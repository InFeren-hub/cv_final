#!/usr/bin/env python3
"""Export training, validation, and zero-shot comparison plots.

The training script writes CSV files with the following names:

- train_metrics.csv
- val_metrics.csv

This utility compares the B-only run and the A+B+C joint run, then writes PNG
figures and summary JSON files used by the report.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def read_metrics(run_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_csv = run_dir / "train_metrics.csv"
    val_csv = run_dir / "val_metrics.csv"
    if not train_csv.exists():
        raise FileNotFoundError(f"missing train metrics: {train_csv}")
    if not val_csv.exists():
        raise FileNotFoundError(f"missing validation metrics: {val_csv}")
    return pd.read_csv(train_csv), pd.read_csv(val_csv)


def plot_metric(
    df: pd.DataFrame,
    y: str,
    title: str,
    ylabel: str,
    path: Path,
    markers: bool = False,
) -> None:
    plt.figure(figsize=(8, 4.8), dpi=160)
    for label in ["B-only", "A+B+C joint"]:
        sub = df[df["run"] == label]
        plt.plot(
            sub["step"],
            sub[y],
            label=label,
            linewidth=2,
            marker="o" if markers else None,
            markersize=4,
        )
    plt.xlabel("Training step")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def maybe_plot_zero_shot(zero_shot_dir: Path) -> None:
    summary_path = zero_shot_dir / "zero_shot_d_summary.json"
    if not summary_path.exists():
        return
    with summary_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    labels = ["B-only", "A+B+C joint"]
    chunk = [
        data["b_only_chunk_action_l1"],
        data["joint_abc_chunk_action_l1"],
    ]
    first = [
        data["b_only_first_action_l1"],
        data["joint_abc_first_action_l1"],
    ]

    x = range(len(labels))
    width = 0.35
    plt.figure(figsize=(7, 4.5), dpi=160)
    plt.bar([i - width / 2 for i in x], chunk, width=width, label="Chunk Action L1")
    plt.bar([i + width / 2 for i in x], first, width=width, label="First-action L1")
    plt.xticks(list(x), labels)
    plt.ylabel("Action L1 error")
    plt.title("Zero-shot action error on CALVIN environment D")
    plt.grid(axis="y", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(zero_shot_dir / "fig_zero_shot_d_action_error.png")
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--b-run", required=True, type=Path)
    parser.add_argument("--joint-run", required=True, type=Path)
    parser.add_argument("--zero-shot-dir", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    b_train, b_val = read_metrics(args.b_run)
    j_train, j_val = read_metrics(args.joint_run)
    b_train["run"] = "B-only"
    b_val["run"] = "B-only"
    j_train["run"] = "A+B+C joint"
    j_val["run"] = "A+B+C joint"

    train = pd.concat([b_train, j_train], ignore_index=True)
    val = pd.concat([b_val, j_val], ignore_index=True)
    train.to_csv(args.output_dir / "comparison_train_metrics.csv", index=False)
    val.to_csv(args.output_dir / "comparison_val_metrics.csv", index=False)

    plot_metric(
        train,
        "loss",
        "Training total loss: B-only vs A+B+C joint",
        "Loss",
        args.output_dir / "fig_train_total_loss.png",
    )
    plot_metric(
        train,
        "l1_loss",
        "Training Action L1 loss: B-only vs A+B+C joint",
        "Action L1 loss",
        args.output_dir / "fig_train_action_l1_loss.png",
    )
    plot_metric(
        val,
        "val/loss",
        "Validation total loss: B-only vs A+B+C joint",
        "Validation loss",
        args.output_dir / "fig_validation_total_loss.png",
        markers=True,
    )
    plot_metric(
        val,
        "val/l1_loss",
        "Validation Action L1 loss: B-only vs A+B+C joint",
        "Validation Action L1 loss",
        args.output_dir / "fig_validation_action_l1_loss.png",
        markers=True,
    )

    summary = {
        "B-only": {
            "final_step": int(b_train["step"].iloc[-1]),
            "final_train_loss": float(b_train["loss"].iloc[-1]),
            "final_train_l1_loss": float(b_train["l1_loss"].iloc[-1]),
            "final_val_loss": float(b_val["val/loss"].iloc[-1]),
            "final_val_l1_loss": float(b_val["val/l1_loss"].iloc[-1]),
        },
        "A+B+C joint": {
            "final_step": int(j_train["step"].iloc[-1]),
            "final_train_loss": float(j_train["loss"].iloc[-1]),
            "final_train_l1_loss": float(j_train["l1_loss"].iloc[-1]),
            "final_val_loss": float(j_val["val/loss"].iloc[-1]),
            "final_val_l1_loss": float(j_val["val/l1_loss"].iloc[-1]),
        },
    }
    with (args.output_dir / "validation_curve_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    if args.zero_shot_dir:
        maybe_plot_zero_shot(args.zero_shot_dir)


if __name__ == "__main__":
    main()
