import argparse
from pathlib import Path
from PIL import Image
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import tifffile
import zarr
from tqdm import tqdm

Image.MAX_IMAGE_PIXELS = None


"""
Conversion pipeline for transforming volumetric TIFF image stacks into Zarr datasets.

The script processes sample folders containing 3D TIFF image stacks and optional
annotation images (labels and masks). TIFF slices are read individually and
written directly into a chunked Zarr array, avoiding loading the complete
volume into memory. TIFF volume data is copied to Zarr without modifying the
pixel values or dtype. Labels and masks are converted to binary uint8 images
with values 0 and 1. Spatial dimensions are aligned by cropping the volume,
label, and mask data to their common minimum height and width.


For each processed segment, the script creates a Zarr store containing:

    - volume: 3D image data with shape (Z, Y, X)
    - label: 2D binary annotation map with shape (Y, X), if available
    - mask: 2D binary mask with shape (Y, X), if available

Metadata describing the original and processed array shapes as well as the
selected Zarr chunk size is stored in a separate JSON file.

The input directory is expected to contain scroll, split, and segment
subdirectories with the following structure:

    data/
    ├── scroll_1/
    │   ├── labeled/
    │   │   └── segment_1/
    │   │       ├── layers/
    │   │       │   ├── 000.tif
    │   │       │   ├── 001.tif
    │   │       │   └── ...
    │   │       ├── segment_1_inklabels.png   (optional)
    │   │       └── segment_1_mask.png        (optional)
    │   └── unlabeled/
    │       └── ...
    └── scroll_2/
        └── ...

The output directory preserves the scroll, split, and segment directory
structure. Each processed segment contains its metadata file and Zarr store:

    data/
    └── zarr_data/
        ├── scroll_1/
        │   ├── labeled/
        │   │   └── segment_1/
        │   │       ├── metadata.json
        │   │       └── segment_1.zarr/
        │   └── unlabeled/
        │       └── ...
        └── scroll_2/
            └── ...

Usage:
    python tiff_to_zarr.py <input_dir> <output_dir>

Example:
    python tiff_to_zarr.py ./tiff_data ./zarr_data

Optional arguments:
    --segments SEGMENT [SEGMENT ...]
        Convert only the specified segment folders.
        By default, all segments found in the input directory are processed.

    --chunks Z Y X
        Define the Zarr chunk size for the volumetric image data.
        The default is 1 x 512 x 512.

Example with optional arguments:
    python tiff_to_zarr.py ./tiff_data ./zarr_data \
        --segments segment_1 segment_2 \
        --chunks 1 512 512
"""



def _get_arguments():
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
        default=(1, 512, 512),
        metavar=("Z", "Y", "X"),
        help="Zarr chunk size (default: 1 512 512)",
    )

    args = parser.parse_args()

    return args


def _get_volume_files(folder):
    layers = folder / "layers"

    files = sorted(layers.glob("*.tif"))

    if not files:
        print(f"No TIFF files found in folder: {folder.name}")
        return None

    return files


def _get_label(folder):
    label_path = folder / f"{folder.name}_inklabels.png"

    if not label_path.exists():
        print(f"No label file found for folder: {folder.name}")
        return None
    
    label = _load_binary_image(label_path)

    return label


def _get_mask(folder):
    mask_path = folder / f"{folder.name}_mask.png"
    
    if not mask_path.exists():
        print(f"No mask file found for folder: {folder.name}")
        return None
    
    mask = _load_binary_image(mask_path)

    return mask


def _get_validation_mask(folder):
    validation_mask_path = folder / f"{folder.name}_validation_mask.png"
    
    if not validation_mask_path.exists():
        print(f"No mask file found for folder: {folder.name}")
        return None
    
    validation_mask = _load_binary_image(validation_mask_path)

    return validation_mask


def _load_binary_image(path):
    image = np.array(Image.open(path).convert("L"))
    binary_image = (image > 127).astype(np.uint8)
    return binary_image


def _process_folder(folder, output_root, chunks):
    print(f"Processing folder: {folder.name}")
    
    # output_path = args.output / folder.parent.name / folder.name
    output_path = output_root / folder.parent.parent.name / folder.parent.name / folder.name

    if output_path.exists():
        print(f"Folder already exists for segment: {folder.name}. Skipping.")
        return

    # volume_original = _get_volume(folder)
    volume_files = _get_volume_files(folder)
    label_original = _get_label(folder)
    mask_original = _get_mask(folder)
    validation_mask_original = _get_validation_mask(folder)

    # if volume_original is None:
    #     print(f"Skipping folder {folder.name} due to missing image stack.")
    #     continue

    if volume_files is None:
        print(f"Skipping folder {folder.name} due to missing image stack.")
        return

    output_path.mkdir(parents=True, exist_ok=False)

    first_slice = tifffile.imread(volume_files[0])
    d = len(volume_files)
    h = first_slice.shape[0]
    w = first_slice.shape[1]

    # volume, label, mask = _crop_images(volume_original, label_original, mask_original)

    arrays = [first_slice]

    if label_original is not None:
        arrays.append(label_original)

    if mask_original is not None:
        arrays.append(mask_original)

    if validation_mask_original is not None:
        arrays.append(validation_mask_original)

    h_min = min(a.shape[-2] for a in arrays)
    w_min = min(a.shape[-1] for a in arrays)

    volume_shape = (d, h_min, w_min)

    label = None if label_original is None else label_original[:h_min, :w_min]
    mask = None if mask_original is None else mask_original[:h_min, :w_min]
    validation_mask = None if validation_mask_original is None else validation_mask_original[:h_min, :w_min]

    meta = {
        "volume_original_shape": [d, h, w],
        "label_original_shape": None if label_original is None else label_original.shape,
        "mask_original_shape": None if mask_original is None else mask_original.shape,
        "validation_mask_original_shape": None if validation_mask_original is None else validation_mask_original.shape,
        "volume_shape": volume_shape,
        "label_shape": None if label is None else label.shape,
        "mask_shape": None if mask is None else mask.shape,
        "validation_mask_shape": None if validation_mask is None else validation_mask.shape,
        "chunks": chunks
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
        shape=volume_shape,
        chunks=chunks,
        dtype=first_slice.dtype,
    )

    # z["volume"][:] = volume

    volume_ds = z["volume"]

    # for i, file in tqdm(enumerate(volume_files), desc="Loading and saving TIFFs", total=len(volume_files)):
    for i, file in enumerate(volume_files):
        img = tifffile.imread(file)
        volume_ds[i] = img[:h_min, :w_min]


    # Save label

    if label is not None:
        z.create_array(
            "label",
            shape=label.shape,
            chunks=chunks[1:],
            dtype=label.dtype,
        )

        z["label"][:] = label
    # else:
    #     print("No label")


    # Save mask

    if mask is not None:
        z.create_array(
            "mask",
            shape=mask.shape,
            chunks=chunks[1:],
            dtype=mask.dtype,
        )

        z["mask"][:] = mask
    # else:
    #     print("No mask")


    # Save validation mask

    if validation_mask is not None:
        z.create_array(
            "validation_mask",
            shape=validation_mask.shape,
            chunks=chunks[1:],
            dtype=validation_mask.dtype,
        )

        z["validation_mask"][:] = validation_mask
    # else:
    #     print("No validation mask")

    # print(f"Saved Zarr archive to: {output_path}")
    # print("")

    return f"Zarr archive saved to: {output_path} (label: {label is not None}, mask: {mask is not None}, validation mask: {validation_mask is not None})"


def main(args):
    if args.input.resolve() == args.output.resolve():
        raise ValueError("Input and output directory must be different.")

    args.output.mkdir(parents=True, exist_ok=True)

    # segment_roots = [p for p in args.input.iterdir() if p.is_dir()]

    # if args.segments:
    #     folders = []
    #     for segment in args.segments:
    #         found = False

    #         for root in segment_roots:
    #             segment_path = root / segment

    #             if segment_path.is_dir():
    #                 folders.append(segment_path)
    #                 found = True

    #         if not found:
    #             print(f"Warning: Segment '{segment}' was not found in any subfolder of '{args.input}'. Skipping.")
    # else:
    #     folders = []
    #     for root in segment_roots:
    #         folders.extend([p for p in root.iterdir() if p.is_dir()])

    folders = []
    for scroll in args.input.iterdir():
        for split in scroll.iterdir():
            for segment in split.iterdir():
                if args.segments is None or segment.name in args.segments:
                    folders.append(segment)

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = []
        
        for folder in folders:
            futures.append(executor.submit(
                _process_folder,
                folder,
                args.output,
                args.chunks
            ))

        for future in as_completed(futures):
            print(future.result())


if __name__ == "__main__":
    main(_get_arguments())
