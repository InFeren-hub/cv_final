#!/usr/bin/env python3
import argparse
import csv
import datetime as dt
import io
import json
import os
import time
from bisect import bisect_right
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault('HF_HOME', '/root/autodl-tmp/hf_cache')
os.environ.setdefault('HF_DATASETS_CACHE', '/root/autodl-tmp/hf_cache/datasets')
os.environ.setdefault('TORCH_HOME', '/root/autodl-tmp/torch_cache')
os.environ.setdefault('WANDB_DISABLE_CODE', 'true')

import pyarrow.parquet as pq
import torch
import wandb
from PIL import Image
from torch.amp import GradScaler
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from torchvision import transforms

from lerobot.configs.default import DatasetConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.compute_stats import aggregate_stats
from lerobot.datasets.factory import IMAGENET_STATS
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.datasets.utils import cycle
from lerobot.optim.factory import make_optimizer_and_scheduler
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.factory import make_policy
from lerobot.scripts.train import update_policy
from lerobot.utils.logging_utils import AverageMeter, MetricsTracker
from lerobot.utils.random_utils import set_seed
from lerobot.utils.train_utils import get_step_checkpoint_dir, save_checkpoint, update_last_checkpoint

BASE = Path('/root/autodl-tmp/cv_hw3_task2')
DATA = BASE / 'data/calvin-lerobot'
TO_TENSOR = transforms.ToTensor()


def decode_image(value):
    if isinstance(value, dict):
        data = value.get('bytes')
        if data is None and value.get('path'):
            return TO_TENSOR(Image.open(value['path']).convert('RGB'))
        return TO_TENSOR(Image.open(io.BytesIO(data)).convert('RGB'))
    return TO_TENSOR(value.convert('RGB'))


class CalvinParquetDataset(Dataset):
    def __init__(self, root: Path, episode_indices=None, action_chunk_size=64, cache_size=8):
        self.root = Path(root)
        self.action_chunk_size = action_chunk_size
        self.cache_size = cache_size
        self.cache = OrderedDict()
        allowed = set(episode_indices) if episode_indices is not None else None
        self.episodes = []
        with (self.root / 'meta/episodes.jsonl').open() as f:
            for line in f:
                rec = json.loads(line)
                ep = int(rec['episode_index'])
                if allowed is not None and ep not in allowed:
                    continue
                length = int(rec['length'])
                path = self.root / 'data' / f'chunk-{ep // 1000:03d}' / f'episode_{ep:06d}.parquet'
                self.episodes.append({'episode_index': ep, 'length': length, 'path': path})
        self.cumulative = []
        total = 0
        for rec in self.episodes:
            total += rec['length']
            self.cumulative.append(total)
        self.num_frames = total
        self.num_episodes = len(self.episodes)

    def __len__(self):
        return self.num_frames

    def _load_episode(self, ep_pos):
        if ep_pos in self.cache:
            self.cache.move_to_end(ep_pos)
            return self.cache[ep_pos]
        table = pq.read_table(self.episodes[ep_pos]['path']).to_pydict()
        self.cache[ep_pos] = table
        if len(self.cache) > self.cache_size:
            self.cache.popitem(last=False)
        return table

    def __getitem__(self, idx):
        ep_pos = bisect_right(self.cumulative, idx)
        ep_start = 0 if ep_pos == 0 else self.cumulative[ep_pos - 1]
        frame = idx - ep_start
        ep = self.episodes[ep_pos]
        table = self._load_episode(ep_pos)
        length = ep['length']
        query = [min(frame + i, length - 1) for i in range(self.action_chunk_size)]
        action_is_pad = [frame + i >= length for i in range(self.action_chunk_size)]
        return {
            'observation.images.image': decode_image(table['observation.images.image'][frame]),
            'observation.images.wrist_image': decode_image(table['observation.images.wrist_image'][frame]),
            'observation.state': torch.tensor(table['observation.state'][frame], dtype=torch.float32),
            'action': torch.tensor([table['action'][j] for j in query], dtype=torch.float32),
            'action_is_pad': torch.tensor(action_is_pad, dtype=torch.bool),
        }


def split_episodes(root: Path, val_ratio=0.1, fixed_train_episodes=None):
    eps = []
    with (root / 'meta/episodes.jsonl').open() as f:
        for line in f:
            eps.append(int(json.loads(line)['episode_index']))
    eps = sorted(eps)
    if fixed_train_episodes is not None:
        train = eps[:fixed_train_episodes]
        val = eps[fixed_train_episodes:]
    else:
        n_val = max(1, int(round(len(eps) * val_ratio)))
        train = eps[:-n_val]
        val = eps[-n_val:]
    return train, val


def build_run(mode):
    split_names = ['splitB'] if mode == 'b_only' else ['splitA', 'splitB', 'splitC']
    train_datasets, val_datasets, metas, train_stats = [], [], [], []
    split_info = {}
    for split in split_names:
        root = DATA / split
        fixed = 5500 if mode == 'b_only' and split == 'splitB' else None
        train_eps, val_eps = split_episodes(root, fixed_train_episodes=fixed)
        meta = LeRobotDatasetMetadata(f'local/calvin_{split}', root=root)
        metas.append(meta)
        train_datasets.append(CalvinParquetDataset(root, train_eps))
        val_datasets.append(CalvinParquetDataset(root, val_eps))
        for ep in train_eps:
            train_stats.append(meta.episodes_stats[ep])
        split_info[split] = {'train_episodes': len(train_eps), 'val_episodes': len(val_eps)}
    combined_stats = aggregate_stats(train_stats)
    for image_key in ['observation.images.image', 'observation.images.wrist_image']:
        combined_stats.setdefault(image_key, {})
        for stats_type, value in IMAGENET_STATS.items():
            combined_stats[image_key][stats_type] = torch.tensor(value, dtype=torch.float32)
    meta = SimpleNamespace(
        features=metas[0].features,
        stats=combined_stats,
        fps=metas[0].fps,
        total_episodes=sum(d.num_episodes for d in train_datasets),
        total_frames=sum(d.num_frames for d in train_datasets),
    )
    return ConcatDataset(train_datasets), ConcatDataset(val_datasets), meta, split_info


def tensor_to_float(value):
    if isinstance(value, torch.Tensor):
        return value.detach().float().mean().item()
    return float(value)


def move_batch(batch, device):
    for key, value in list(batch.items()):
        if isinstance(value, torch.Tensor):
            batch[key] = value.to(device, non_blocking=True)
    return batch


def evaluate(policy, loader, device, max_batches):
    policy.train()  # ACT VAE loss needs train-mode latent outputs; no_grad keeps validation read-only.
    total_loss = total_l1 = total_kld = 0.0
    count = 0
    start = time.perf_counter()
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if max_batches is not None and i >= max_batches:
                break
            batch = move_batch(batch, device)
            loss, out = policy.forward(batch)
            total_loss += loss.item()
            total_l1 += float(out.get('l1_loss', 0.0))
            total_kld += float(out.get('kld_loss', 0.0))
            count += 1
    return {
        'val/loss': total_loss / max(count, 1),
        'val/l1_loss': total_l1 / max(count, 1),
        'val/kld_loss': total_kld / max(count, 1),
        'val/batches': count,
        'val/eval_s': time.perf_counter() - start,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['b_only', 'joint_abc'], required=True)
    ap.add_argument('--steps', type=int, default=12000)
    ap.add_argument('--batch-size', type=int, default=64)
    ap.add_argument('--num-workers', type=int, default=8)
    ap.add_argument('--log-freq', type=int, default=50)
    ap.add_argument('--val-freq', type=int, default=1000)
    ap.add_argument('--max-val-batches', type=int, default=128)
    ap.add_argument('--run-name', default=None)
    args = ap.parse_args()

    set_seed(1000)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device('cuda')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for retraining; current instance has no GPU.')

    run_name = args.run_name or f"act_{args.mode}_bs{args.batch_size}_valcurve_{dt.datetime.now():%Y%m%d_%H%M%S}"
    out_dir = BASE / 'outputs' / run_name
    train_ds, val_ds, meta, split_info = build_run(args.mode)

    policy_cfg = ACTConfig(device='cuda', push_to_hub=False, chunk_size=64, n_action_steps=64)
    cfg = TrainPipelineConfig(
        dataset=DatasetConfig(repo_id=f'local/calvin_{args.mode}', root=str(DATA)),
        policy=policy_cfg,
        output_dir=out_dir,
        job_name=run_name,
        seed=1000,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        steps=args.steps,
        eval_freq=0,
        log_freq=args.log_freq,
        save_checkpoint=True,
        save_freq=args.steps,
        wandb=None,
    )
    cfg.validate()
    out_dir.mkdir(parents=True, exist_ok=True)

    policy = make_policy(cfg.policy, ds_meta=meta)
    optimizer, lr_scheduler = make_optimizer_and_scheduler(cfg, policy)
    grad_scaler = GradScaler(device.type, enabled=cfg.policy.use_amp)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False, pin_memory=True)
    train_iter = cycle(train_loader)

    wb = wandb.init(
        project='calvin-act-val-curves',
        entity='escan0r-fudan',
        name=run_name,
        job_type='retrain-with-validation',
        config={
            'mode': args.mode,
            'steps': args.steps,
            'batch_size': args.batch_size,
            'chunk_size': 64,
            'n_action_steps': 64,
            'lr': 1e-5,
            'optimizer': 'AdamW',
            'val_freq': args.val_freq,
            'max_val_batches': args.max_val_batches,
            'splits': split_info,
        },
    )

    train_csv = out_dir / 'train_metrics.csv'
    val_csv = out_dir / 'val_metrics.csv'
    with train_csv.open('w', newline='') as tf, val_csv.open('w', newline='') as vf:
        train_writer = csv.DictWriter(tf, fieldnames=['step','loss','l1_loss','kld_loss','grad_norm','lr','dataloading_s','update_s'])
        val_writer = csv.DictWriter(vf, fieldnames=['step','val/loss','val/l1_loss','val/kld_loss','val/batches','val/eval_s'])
        train_writer.writeheader(); val_writer.writeheader()
        metrics = {
            'loss': AverageMeter('loss', ':.3f'),
            'grad_norm': AverageMeter('grdn', ':.3f'),
            'lr': AverageMeter('lr', ':0.1e'),
            'update_s': AverageMeter('updt_s', ':.3f'),
            'dataloading_s': AverageMeter('data_s', ':.3f'),
        }
        tracker = MetricsTracker(args.batch_size, meta.total_frames, meta.total_episodes, metrics, initial_step=0)
        print(f'START {run_name} mode={args.mode} train_frames={meta.total_frames} train_episodes={meta.total_episodes} val_frames={len(val_ds)} val_episodes={sum(d.num_episodes for d in val_ds.datasets)} split_info={split_info}', flush=True)
        for step in range(1, args.steps + 1):
            start = time.perf_counter()
            batch = next(train_iter)
            tracker.dataloading_s = time.perf_counter() - start
            batch = move_batch(batch, device)
            tracker, out = update_policy(tracker, policy, batch, optimizer, cfg.optimizer.grad_clip_norm, grad_scaler=grad_scaler, lr_scheduler=lr_scheduler, use_amp=cfg.policy.use_amp)
            tracker.step()
            if step % args.log_freq == 0 or step == 1:
                row = {
                    'step': step,
                    'loss': tracker.loss.avg,
                    'l1_loss': float(out.get('l1_loss', 0.0)),
                    'kld_loss': float(out.get('kld_loss', 0.0)),
                    'grad_norm': tracker.grad_norm.avg,
                    'lr': tracker.lr.avg,
                    'dataloading_s': tracker.dataloading_s.avg,
                    'update_s': tracker.update_s.avg,
                }
                train_writer.writerow(row); tf.flush()
                wandb.log({f'train/{k}': v for k, v in row.items() if k != 'step'}, step=step)
                print(f"step={step} loss={row['loss']:.6f} l1={row['l1_loss']:.6f} data_s={row['dataloading_s']:.3f} update_s={row['update_s']:.3f}", flush=True)
                tracker.reset_averages()
            if step % args.val_freq == 0 or step == args.steps:
                val = evaluate(policy, val_loader, device, args.max_val_batches)
                val_row = {'step': step, **val}
                val_writer.writerow(val_row); vf.flush()
                wandb.log(val, step=step)
                print(f"VAL step={step} loss={val['val/loss']:.6f} l1={val['val/l1_loss']:.6f} batches={val['val/batches']} eval_s={val['val/eval_s']:.1f}", flush=True)
        checkpoint_dir = get_step_checkpoint_dir(cfg.output_dir, cfg.steps, args.steps)
        save_checkpoint(checkpoint_dir, args.steps, cfg, policy, optimizer, lr_scheduler)
        update_last_checkpoint(checkpoint_dir)
        print(f'SAVED {checkpoint_dir}', flush=True)
    wb.finish()
    print(f'DONE {run_name} out_dir={out_dir}', flush=True)

if __name__ == '__main__':
    main()
