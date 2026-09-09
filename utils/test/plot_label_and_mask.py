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

    parser.add_argument(
        "--margin",
        default=50,
        type=int,
        help="Margin",
    )

    return parser.parse_args()


def main(args):
    label = np.array(Image.open(args.label).convert("L"))

    # mask = create_training_mask(label, margin=args.margin)
    mask = create_training_mask(label, margin_factor=args.margin)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    axes[0].imshow(label, cmap="gray")
    axes[0].set_title("Label")
    axes[0].axis("off")

    axes[1].imshow(mask, cmap="gray")
    axes[1].set_title(f"Training Mask (margin={args.margin})")
    axes[1].axis("off")

    plt.tight_layout()
    # plt.show()

    plt.savefig("outputs/plot.png")


if __name__ == "__main__":
    main(_get_arguments())
