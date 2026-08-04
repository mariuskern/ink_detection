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
        "data_tiff",
        type=Path,
    )

    parser.add_argument(
        "data_zarr",
        type=Path,
    )

    return parser.parse_args()


def main(args):
    z = zarr.open(args.data_zarr / f"{args.data_tiff.name}.zarr", mode="r")
    volume = np.array(z["volume"])

    d, h, w = volume.shape


    # Volume

    files = sorted((args.data_tiff / "layers").glob("*.tif"))
    t = np.stack([tifffile.imread(f) for f in files])

    print(np.array_equal(volume, t[:d, :h, :w]))


    # Label

    label_path = args.data_tiff / f"{args.data_tiff.name}_inklabels.png"
    if label_path.exists():
        label = np.array(Image.open(label_path).convert("L"))
        label = (label > 127).astype(np.uint8)
        z_label = np.array(z["label"])
        print(np.array_equal(z_label, label[:h, :w]))


    # Maks

    mask_path = args.data_tiff / f"{args.data_tiff.name}_mask.png"
    if mask_path.exists():
        mask = np.array(Image.open(mask_path).convert("L"))
        mask = (mask > 127).astype(np.uint8)
        z_mask = np.array(z["mask"])
        print(np.array_equal(z_mask, mask[:h, :w]))


if __name__ == "__main__":
    main(_get_arguments())
