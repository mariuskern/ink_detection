import argparse
from pathlib import Path
from PIL import Image
import json

import numpy as np
import tifffile
import zarr

Image.MAX_IMAGE_PIXELS = None


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


def _get_volume(folder):
    layers = folder / "layers"

    files = sorted(layers.glob("*.tif"))

    if not files:
        print(f"No TIFF files found in folder: {folder.name}")
        return None
    
    volume = np.stack([tifffile.imread(f) for f in files])

    return volume

def _get_label(folder):
    label_path = folder / f"{folder.name}_inklabels.png"

    if not label_path.exists():
        print(f"No label file found for folder: {folder.name}")
        return None
    
    label = _load_binary_image(label_path)

    return label

def _get_mask(folder):
    mask_path = folder / f"{folder.name}_inklabels.png"
    
    if not mask_path.exists():
        print(f"No mask file found for folder: {folder.name}")
        return None
    
    mask = _load_binary_image(mask_path)

    return mask

def _load_binary_image(path):
    image = np.array(Image.open(path).convert("L"))
    binary_image = (image > 127).astype(np.uint8)
    return binary_image

def _crop_images(volume, label=None, mask=None):
    arrays = [volume]

    if label is not None:
        arrays.append(label)
    
    if mask is not None:
        arrays.append(mask)

    h_min = min(arr.shape[-2] for arr in arrays)
    w_min = min(arr.shape[-1] for arr in arrays)

    volume = volume[:, :h_min, :w_min]
    label = None if label is None else label[:h_min, :w_min]
    mask = None if mask is None else mask[:h_min, :w_min]

    return volume, label, mask

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

        output_path = args.output / folder.name

        if output_path.exists():
            print(f"Zarr archive already exists for folder: {folder.name}. Skipping.")
            continue
        
        output_path.mkdir(parents=True)

        # z = zarr.open(
        #     output_path,
        #     mode="w",
        #     shape=stack.shape,
        #     chunks=args.chunks,
        #     dtype=stack.dtype,
        # )

        volume_original = _get_volume(folder)
        label_original = _get_label(folder)
        mask_original = _get_mask(folder)

        if volume_original is None:
            print(f"Skipping folder {folder.name} due to missing image stack.")
            continue

        volume, label, mask = _crop_images(volume_original, label_original, mask_original)

        meta = {
            "volume_original_shape": volume_original.shape,
            "label_original_shape": None if label_original is None else label_original.shape,
            "mask_original_shape": None if mask_original is None else mask_original.shape,
            "volume_shape": volume.shape,
            "label_shape": None if label is None else label.shape,
            "mask_shape": None if mask is None else mask.shape,
            "chunks": args.chunks
        }

        with open(output_path / "metadata.json", "w") as f:
            json.dump(meta, f, indent=4)

        z = zarr.open(
            output_path / f"{folder.name}.zarr",
            mode="w"
        )


        # Save volume

        z.create_array(
            "volume",
            shape=volume.shape,
            chunks=args.chunks,
            dtype=volume.dtype,
        )

        z["volume"][:] = volume


        # Save label

        if label is not None:
            z.create_array(
                "label",
                shape=label.shape,
                chunks=args.chunks[1:],
                dtype=label.dtype,
            )

            z["label"][:] = label


        # Save mask

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
