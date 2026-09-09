import argparse
from pathlib import Path
from PIL import Image

import numpy as np


Image.MAX_IMAGE_PIXELS = None


def _get_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "png",
        type=Path,
        help="Path to png",
    )

    return parser.parse_args()


def main(args):
    image = np.array(Image.open(args.png))
    print(np.unique(image))


if __name__ == "__main__":
    main(_get_arguments())