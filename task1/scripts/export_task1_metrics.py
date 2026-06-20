from pathlib import Path
import csv
import os

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import matplotlib.pyplot as plt


ROOT = Path(r"D:\计算机视觉HW3")
OUT = ROOT / "task1" / "reports" / "metrics"
OUT.mkdir(parents=True, exist_ok=True)
WANDB_DIR = OUT / "wandb_offline"
WANDB_DIR.mkdir(parents=True, exist_ok=True)


SOURCES = {
    "B_text_to_3D": Path(r"D:\cvhw3\task1\third_party\threestudio\outputs\object-b-magic3d-coarse-sd\magic3d_robot_coarse@20260618-234612\tb_logs\version_0"),
    "C_single_image": Path(r"D:\cvhw3\task1\third_party\threestudio\outputs\magic123-coarse-sd\object_C@20260602-145930\tb_logs\version_0"),
    "Garden_2DGS": Path(r"D:\cvhw3\task1\2d-gaussian-splatting\output\background_garden"),
}


PLOTS = {
    "B_text_to_3D": ["train/loss_sds", "train/loss_orient", "train/loss_sparsity", "train/loss_opaque"],
    "C_single_image": ["train/loss_rgb", "train/loss_mask", "train/loss_sd", "train/loss_sd_3d"],
    "Garden_2DGS": ["train_loss_patches/total_loss", "train_loss_patches/reg_loss", "total_points"],
}


def load_scalars(path):
    ea = EventAccumulator(str(path), size_guidance={"scalars": 0})
    ea.Reload()
    data = {}
    for tag in ea.Tags().get("scalars", []):
        data[tag] = ea.Scalars(tag)
    return data


def write_csv(name, data, tags):
    csv_path = OUT / f"{name}_scalars.csv"
    rows = []
    for tag in tags:
        for s in data.get(tag, []):
            rows.append([tag, s.step, s.value, s.wall_time])
    rows.sort(key=lambda r: (r[0], r[1]))
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["tag", "step", "value", "wall_time"])
        writer.writerows(rows)
    return csv_path


def plot_tags(name, data, tags):
    fig, ax = plt.subplots(figsize=(8.8, 4.8), dpi=160)
    for tag in tags:
        vals = data.get(tag, [])
        if not vals:
            continue
        xs = [s.step for s in vals]
        ys = [s.value for s in vals]
        if len(xs) > 1200:
            stride = max(len(xs) // 1200, 1)
            xs = xs[::stride]
            ys = ys[::stride]
        label = tag.replace("train/", "").replace("train_loss_patches/", "")
        ax.plot(xs, ys, linewidth=1.2, label=label)
    ax.set_title(name.replace("_", " ") + " training curves")
    ax.set_xlabel("Step / Iteration")
    ax.set_ylabel("Value")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    png = OUT / f"{name}_curves.png"
    fig.savefig(png)
    plt.close(fig)
    return png


def summarize_runtime(data, preferred_tag):
    vals = data.get(preferred_tag, [])
    if len(vals) < 2:
        return None
    hours = (vals[-1].wall_time - vals[0].wall_time) / 3600.0
    return hours


def log_wandb_offline(all_data):
    try:
        import wandb
    except Exception as exc:
        return f"wandb unavailable: {exc}"

    os.environ["WANDB_MODE"] = "offline"
    os.environ["WANDB_DIR"] = str(WANDB_DIR)
    run = wandb.init(project="cv_hw3_task1", name="task1_metrics_export", dir=str(WANDB_DIR), reinit=True)
    for source_name, data in all_data.items():
        tags = PLOTS[source_name]
        max_len = max((len(data.get(t, [])) for t in tags), default=0)
        stride = max(max_len // 800, 1)
        # Log a compact subset so the offline run remains small.
        for i in range(0, max_len, stride):
            payload = {}
            step = None
            for tag in tags:
                vals = data.get(tag, [])
                if i < len(vals):
                    step = vals[i].step
                    payload[f"{source_name}/{tag}"] = vals[i].value
            if payload and step is not None:
                run.log(payload, step=int(step))
    run.finish()
    return str(WANDB_DIR)


def main():
    all_data = {}
    outputs = []
    for name, path in SOURCES.items():
        data = load_scalars(path)
        all_data[name] = data
        csv_path = write_csv(name, data, PLOTS[name])
        png_path = plot_tags(name, data, PLOTS[name])
        outputs.append((name, csv_path, png_path))

    wandb_result = log_wandb_offline(all_data)
    summary = OUT / "metrics_summary.txt"
    with summary.open("w", encoding="utf-8") as f:
        f.write("Task1 metrics export\n")
        f.write(f"WandB offline directory: {wandb_result}\n\n")
        for name, data in all_data.items():
            tag = PLOTS[name][0]
            runtime = summarize_runtime(data, tag)
            last_values = []
            for t in PLOTS[name]:
                vals = data.get(t, [])
                if vals:
                    last_values.append(f"{t}={vals[-1].value:.6g} @ step {vals[-1].step}")
            f.write(f"[{name}]\n")
            if runtime is not None:
                f.write(f"event_wall_time_hours={runtime:.2f}\n")
            f.write("\n".join(last_values) + "\n\n")
    for item in outputs:
        print(item[0], item[1], item[2])
    print(summary)
    print("wandb", wandb_result)


if __name__ == "__main__":
    main()
