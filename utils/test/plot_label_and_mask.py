import os
import sys
import argparse
from pathlib import Path
from PIL import Image

import matplotlib.pyplot as plt
import numpy as np

Image.MAX_IMAGE_PIXELS = None

from dotenv import load_dotenv

load_dotenv()
sys.path.append(os.getenv("CODE_DIR"))

from utils.create_training_mask import create_training_mask


def _get_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "label",
        type=Path,
        help="Label",
    )

    margins = parser.add_mutually_exclusive_group()

    margins.add_argument(
        "--margin",
        default=None,
        type=int,
        help="Fixed Euclidean radius in pixels (blockwise processing)",
    )

    margins.add_argument(
        "--margin-factor",
        default=0.2,
        type=float,
        help="Uniform margin relative to median component size (default: 0.2)",
    )

    parser.add_argument(
        "--chunk-size",
        default=2048,
        type=int,
    )

    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress bars",
    )

    closing = parser.add_mutually_exclusive_group()

    closing.add_argument(
        "--closing-radius",
        default=None,
        type=int,
        help="Optional gap-closing radius in pixels (0 disables closing)",
    )

    closing.add_argument(
        "--closing-factor",
        default=None,
        type=float,
        help="Gap-closing radius relative to median component size",
    )
    
    parser.add_argument(
        "--closing-method",
        choices=("auto", "distance", "opencv"),
        default="auto",
        help="auto uses distance transforms for radii of 100 or more",
    )

    return parser.parse_args()


def main(args):
    import matplotlib

    matplotlib.use("TkAgg", force=True)
    import matplotlib.pyplot as plt

    label = np.array(Image.open(args.label).convert("L"))

    mask = create_training_mask(
        label,
        margin_factor=args.margin_factor,
        margin=args.margin,
        chunk_size=args.chunk_size,
        closing_radius=args.closing_radius,
        closing_factor=args.closing_factor,
        closing_method=args.closing_method,
        progress=not args.no_progress,
    )

    max_size = 2000
    height, width = label.shape
    scale = min(1.0, max_size / max(height, width))

    if scale < 1:
        size = (max(1, round(width * scale)), max(1, round(height * scale)))
        label = np.asarray(Image.fromarray(label).resize(size, Image.Resampling.NEAREST))
        mask = np.asarray(Image.fromarray(mask).resize(size, Image.Resampling.NEAREST))

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)

    axes[0].imshow(label, cmap="gray")
    axes[0].set_title("Label")
    axes[0].axis("off")

    axes[1].imshow(mask, cmap="gray")
    setting = f"margin={args.margin}px" if args.margin is not None else f"factor={args.margin_factor}"

    if args.closing_radius:
        setting += f", closing={args.closing_radius}px"
    elif args.closing_factor:
        setting += f", closing_factor={args.closing_factor}"

    axes[1].set_title(f"Training Mask ({setting})")
    axes[1].axis("off")

    overlay = np.zeros((*label.shape, 3), dtype=np.uint8)
    overlay[..., 0] = (label > 0).astype(np.uint8) * 255
    overlay[..., 2] = mask
    axes[2].imshow(overlay, interpolation="nearest")
    axes[2].set_title("Overlay: Label=red, Mask=blue\nOverlap=magenta")
    axes[2].axis("off")

    plt.tight_layout()
    plt.show()

    # plt.savefig("outputs/plot.png")


if __name__ == "__main__":
    main(_get_arguments())
