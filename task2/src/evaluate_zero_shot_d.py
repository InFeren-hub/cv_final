#!/usr/bin/env python3
import argparse
import csv
import io
import json
import math
import os
import time
from bisect import bisect_right
from collections import OrderedDict
from pathlib import Path

os.environ.setdefault('TORCH_HOME', '/root/autodl-tmp/torch_cache')
os.environ.setdefault('HF_HOME', '/root/autodl-tmp/hf_cache')
os.environ.setdefault('HF_DATASETS_CACHE', '/root/autodl-tmp/hf_cache/datasets')

import pyarrow.parquet as pq
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms

from lerobot.policies.act.modeling_act import ACTPolicy

TO_TENSOR = transforms.ToTensor()


def decode_image(value):
    if isinstance(value, dict):
        data = value.get('bytes')
        if data is None and value.get('path'):
            return TO_TENSOR(Image.open(value['path']).convert('RGB'))
        return TO_TENSOR(Image.open(io.BytesIO(data)).convert('RGB'))
    return TO_TENSOR(value.convert('RGB'))


class CalvinParquetEvalDataset(Dataset):
    def __init__(self, root: Path, action_chunk_size=64, cache_size=8):
        self.root = Path(root)
        self.action_chunk_size = action_chunk_size
        self.cache_size = cache_size
        self.cache = OrderedDict()
        self.episodes = []
        with (self.root / 'meta' / 'episodes.jsonl').open() as f:
            for line in f:
                rec = json.loads(line)
                ep = int(rec['episode_index'])
                length = int(rec['length'])
                path = self.root / 'data' / f'chunk-{ep // 1000:03d}' / f'episode_{ep:06d}.parquet'
                self.episodes.append({'episode_index': ep, 'length': length, 'path': path})
        total = 0
        self.cumulative = []
        for ep in self.episodes:
            total += ep['length']
            self.cumulative.append(total)
        self.num_frames = total
        self.num_episodes = len(self.episodes)

    def __len__(self):
        return self.num_frames

    def _load_episode(self, ep_pos):
        cached = self.cache.get(ep_pos)
        if cached is not None:
            self.cache.move_to_end(ep_pos)
            return cached
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
        img_key = 'observation.images.image' if 'observation.images.image' in table else 'image'
        wrist_key = 'observation.images.wrist_image' if 'observation.images.wrist_image' in table else 'wrist_image'
        state_key = 'observation.state' if 'observation.state' in table else 'state'
        action_key = 'action' if 'action' in table else 'actions'
        query = [min(frame + i, length - 1) for i in range(self.action_chunk_size)]
        pad = [frame + i >= length for i in range(self.action_chunk_size)]
        return {
            'observation.images.image': decode_image(table[img_key][frame]),
            'observation.images.wrist_image': decode_image(table[wrist_key][frame]),
            'observation.state': torch.tensor(table[state_key][frame], dtype=torch.float32),
            'action': torch.tensor([table[action_key][j] for j in query], dtype=torch.float32),
            'action_is_pad': torch.tensor(pad, dtype=torch.bool),
        }


def evaluate_one(name, ckpt, loader, device, max_batches=None):
    policy = ACTPolicy.from_pretrained(ckpt, local_files_only=True)
    policy.to(device)
    policy.eval()
    total_abs = 0.0
    total_count = 0
    first_abs = 0.0
    first_count = 0
    per_dim_abs = None
    total_batches = 0
    start = time.time()
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            gt = batch['action'].to(device, non_blocking=True)
            pad = batch['action_is_pad'].to(device, non_blocking=True)
            obs = {k: v.to(device, non_blocking=True) for k, v in batch.items() if k.startswith('observation.')}
            pred = policy.predict_action_chunk(obs)
            valid = (~pad).unsqueeze(-1)
            abs_err = (pred - gt).abs() * valid
            total_abs += abs_err.sum().item()
            total_count += valid.sum().item() * gt.shape[-1]
            first_abs += (pred[:, 0] - gt[:, 0]).abs().sum().item()
            first_count += gt.shape[0] * gt.shape[-1]
            dim_sum = abs_err.sum(dim=(0, 1)).detach().cpu()
            per_dim_abs = dim_sum if per_dim_abs is None else per_dim_abs + dim_sum
            total_batches += 1
            if total_batches % 100 == 0:
                print(f'{name}: batch={total_batches} raw_l1={total_abs/total_count:.6f}', flush=True)
    elapsed = time.time() - start
    per_dim = (per_dim_abs / (total_count / gt.shape[-1])).tolist() if total_count else []
    return {
        'model': name,
        'checkpoint': str(ckpt),
        'batches': total_batches,
        'frames_eval': total_batches * loader.batch_size,
        'raw_action_l1_chunk_mean': total_abs / total_count,
        'raw_action_l1_first_action': first_abs / first_count,
        'per_dim_raw_action_l1': per_dim,
        'seconds': elapsed,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='/root/autodl-tmp/cv_hw3_task2/data/calvin-lerobot/splitD')
    ap.add_argument('--batch-size', type=int, default=64)
    ap.add_argument('--num-workers', type=int, default=8)
    ap.add_argument('--max-batches', type=int, default=None)
    ap.add_argument('--stride', type=int, default=1, help='Evaluate every Nth frame to speed up full-dataset estimates.')
    ap.add_argument('--out-dir', default='/root/autodl-tmp/cv_hw3_task2/outputs/zero_shot_d')
    args = ap.parse_args()
    base = Path('/root/autodl-tmp/cv_hw3_task2')
    b_ckpt = base / 'outputs/act_splitB_train5500_bs64_wandb_20260615_232413/checkpoints/012000/pretrained_model'
    j_name = (base / 'latest_joint_run.txt').read_text().strip()
    j_ckpt = base / 'outputs' / j_name / 'checkpoints/012000/pretrained_model'
    ds = CalvinParquetEvalDataset(Path(args.root), action_chunk_size=64)
    if args.stride > 1:
        ds = Subset(ds, list(range(0, len(ds), args.stride)))
    loader = DataLoader(ds, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False, pin_memory=True)
    device = torch.device('cuda')
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for name, ckpt in [('B-only', b_ckpt), ('A+B+C joint', j_ckpt)]:
        print(f'Evaluating {name}: {ckpt}', flush=True)
        results.append(evaluate_one(name, ckpt, loader, device, args.max_batches))
    json_path = out_dir / 'zero_shot_d_action_error.json'
    csv_path = out_dir / 'zero_shot_d_action_error.csv'
    json_path.write_text(json.dumps(results, indent=2))
    with csv_path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['model','checkpoint','batches','frames_eval','raw_action_l1_chunk_mean','raw_action_l1_first_action','seconds'])
        writer.writeheader()
        for r in results:
            row = {k: r[k] for k in writer.fieldnames}
            writer.writerow(row)
    print(json.dumps(results, indent=2), flush=True)
    print(f'WROTE {json_path} {csv_path}', flush=True)

if __name__ == '__main__':
    main()
