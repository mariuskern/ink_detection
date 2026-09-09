import os
import argparse
from pathlib import Path
from PIL import Image

import zarr
import numpy as np
import tifffile

from dotenv import load_dotenv
load_dotenv()


def _get_arguments():
    parser = argparse.ArgumentParser(description="Create raster plot from tile index.")
    
    parser.add_argument(
        "tiff_1",
        type=Path,
    )

    parser.add_argument(
        "tiff_2",
        type=Path,
    )

    return parser.parse_args()


def main(args):
    tiff_1 = tifffile.imread(args.tiff_1)
    tiff_2 = tifffile.imread(args.tiff_2)

    print(f"TIFF 1: shape={tiff_1.shape}, dtype={tiff_1.dtype}")
    print(f"TIFF 2: shape={tiff_2.shape}, dtype={tiff_2.dtype}")

    print(f"Identical: {np.array_equal(tiff_1, tiff_2)}")


if __name__ == "__main__":
    main(_get_arguments())
