import argparse
from pathlib import Path
from PIL import Image

import numpy as np
import tifffile
import zarr


"""
Conversion pipeline for transforming TIFF image stacks into Zarr datasets.

The script reads volumetric TIFF data from individual sample folders,
converts the image stacks into NumPy arrays, and stores them as chunked
Zarr arrays for efficient out-of-core processing. Associated annotation
data (labels and masks) are included when available.

Zarr storage enables fast partial loading of large 3D volumes, which is
particularly useful for training deep learning models on large-scale
microscopy datasets.

Usage:
    python tiff_to_zarr.py <input_dir> <output_dir>

Example:
    python tiff_to_zarr.py ./tiff_data ./zarr_data

Optional arguments:
    --segments SEGMENT [SEGMENT ...]
        Convert only the specified sample folders.

    --chunks Z Y X
        Define the Zarr chunk size for the volumetric data.

Example with optional arguments:
    python tiff_to_zarr.py ./tiff_data ./zarr_data \
        --segments sample_1 sample_2 \
        --chunks 65 512 512
"""


def get_arguments():
    parser = argparse.ArgumentParser(
        description="Convert TIFF image stacks in subfolders to Zarr archives"
    )

    parser.add_argument(
        "input",
        type=Path,
        help="Input directory containing tif subfolders",
    )

    parser.add_argument(
        "output",
        type=Path,
        help="Output directory for zarr archives",
    )

    parser.add_argument(
        "--segments",
        nargs="+",
        default=None,
        help="List of subfolder names to process (default: all subfolders)",
    )

    parser.add_argument(
        "--chunks",
        type=int,
        nargs=3,
        default=(65, 512, 512),
        metavar=("Z", "Y", "X"),
        help="Zarr chunk size (default: 65 512 512)",
    )

    args = parser.parse_args()

    return args


def get_volume(folder):
    layers = folder / "layers"

    files = sorted(layers.glob("*.tif"))

    if not files:
        print(f"No TIFF files found in folder: {folder.name}")
        return None
    
    volume = np.stack([tifffile.imread(f) for f in files])

    return volume

def get_label(folder):
    label_path = folder / f"{folder.name}_inklabels.png"

    if not label_path.exists():
        print(f"No label file found for folder: {folder.name}")
        return None
    
    label = np.array(
        Image.open(label_path).convert("L")
    )

    return label

def get_mask(folder):
    mask_path = folder / f"{folder.name}_inklabels.png"
    
    if not mask_path.exists():
        print(f"No mask file found for folder: {folder.name}")
        return None
    
    mask = np.array(
        Image.open(mask_path).convert("L")
    )

    return mask


def main(args):
    args.output.mkdir(
        parents=True,
        exist_ok=True
    )

    if args.segments:
        folders = [args.input / segment for segment in args.segments]
        folders = [f for f in folders if f.is_dir()]
    else:
        folders = [p for p in args.input.iterdir() if p.is_dir()]

    for folder in folders:
        print(f"Processing folder: {folder.name}")

        output_path = args.output / f"{folder.name}.zarr"

        # z = zarr.open(
        #     output_path,
        #     mode="w",
        #     shape=stack.shape,
        #     chunks=args.chunks,
        #     dtype=stack.dtype,
        # )

        # volume

        volume = get_volume(folder)

        if volume is None:
            print(f"Skipping folder {folder.name} due to missing image stack.")
            continue

        z = zarr.open(
            output_path,
            mode="w"
        )

        z.create_array(
            "volume",
            shape=volume.shape,
            chunks=args.chunks,
            dtype=volume.dtype,
        )

        z["volume"][:] = volume


        # label

        label = get_label(folder)

        if label is not None:
            z.create_array(
                "label",
                shape=label.shape,
                chunks=args.chunks[1:],
                dtype=label.dtype,
            )

            z["label"][:] = label


        # mask

        mask = get_mask(folder)

        if mask is not None:
            z.create_array(
                "mask",
                shape=mask.shape,
                chunks=args.chunks[1:],
                dtype=mask.dtype,
            )

            z["mask"][:] = mask

        print(f"Saved Zarr archive to: {output_path.name}")
        print("")


if __name__ == "__main__":
    main(get_arguments())


# PATH = Path("D:/Dokumente/Studium/Masterarbeit/omar/data_fei/new_labeled/20241025145341/layers/")
# OUT = Path("./output.zarr")

# files = sorted(PATH.glob("*.tif"))
# stack = np.stack([tifffile.imread(f) for f in files])

# z = zarr.open(
#     OUT,
#     mode="w",
#     shape=stack.shape,
#     chunks=(65, 512, 512),
#     dtype=stack.dtype,
# )

# z[:] = stack

# z = zarr.open("output.zarr", mode="r")
# arr = np.asarray(z)

# print(np.array_equal(arr, stack))