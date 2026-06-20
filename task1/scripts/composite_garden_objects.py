import argparse
from pathlib import Path

from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--background_dir", required=True)
    parser.add_argument("--overlay_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--frame_step", type=int, default=3)
    parser.add_argument("--max_frames", type=int, default=60)
    return parser.parse_args()


def main():
    args = parse_args()
    background_dir = Path(args.background_dir)
    overlay_dir = Path(args.overlay_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    backgrounds = sorted(background_dir.glob("*.png"))[:: max(args.frame_step, 1)]
    if args.max_frames > 0:
        backgrounds = backgrounds[: args.max_frames]

    for idx, bg_path in enumerate(backgrounds):
        overlay_path = overlay_dir / f"{idx:04d}.png"
        if not overlay_path.exists():
            continue
        bg = Image.open(bg_path).convert("RGBA")
        fg = Image.open(overlay_path).convert("RGBA").resize(bg.size)
        merged = Image.alpha_composite(bg, fg)
        merged.convert("RGB").save(out_dir / f"{idx:04d}.png", quality=95)


if __name__ == "__main__":
    main()
