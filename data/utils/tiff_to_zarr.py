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

A .conversion_complete marker is written only after all slices are saved.
Existing outputs without that marker are rebuilt on the next invocation.

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

    parser.add_argument(
        "--crops",
        nargs="+",
        default=None,
        metavar="SEGMENT:Y_START,Y_END,X_START,X_END",
        help="Optional crops per segment. Format: SEGMENT:Y_START,Y_END,X_START,X_END. Can be specified multiple times.",
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of worker threads"
    )

    args = parser.parse_args()

    return args


def _parse_crops(crop_args):
    if crop_args is None:
        return {}

    crops = {}

    for crop_arg in crop_args:
        try:
            segment, coordinates = crop_arg.split(":", 1)
            values = [int(v) for v in coordinates.split(",")]

            if len(values) != 4:
                raise ValueError

            crop = values

        except ValueError:
            raise ValueError(f"Invalid crop specification: '{crop_arg}'. Expected format: SEGMENT:Y_START,Y_END,X_START,X_END")

        crops.setdefault(segment, []).append(crop)

    return crops


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
        print(f"No validation mask file found for folder: {folder.name}")
        return None
    
    validation_mask = _load_binary_image(validation_mask_path)

    return validation_mask


def _load_binary_image(path):
    image = np.array(Image.open(path).convert("L"))
    binary_image = (image > 127).astype(np.uint8)
    return binary_image


def _process_folder(folder, output_root, chunks, crops=None):
    print(f"Processing folder: {folder.name}")

    if crops is None:
        crops = [None]

    volume_files = _get_volume_files(folder)
    label_original = _get_label(folder)
    mask_original = _get_mask(folder)
    validation_mask_original = _get_validation_mask(folder)

    if volume_files is None:
        print(f"Skipping folder {folder.name} due to missing image stack.")
        return

    first_slice = tifffile.imread(volume_files[0])
    d = len(volume_files)
    h = first_slice.shape[0]
    w = first_slice.shape[1]

    arrays = [first_slice]

    if label_original is not None:
        arrays.append(label_original)

    if mask_original is not None:
        arrays.append(mask_original)

    if validation_mask_original is not None:
        arrays.append(validation_mask_original)

    h_min = min(a.shape[-2] for a in arrays)
    w_min = min(a.shape[-1] for a in arrays)

    label_base = None if label_original is None else label_original[:h_min, :w_min]
    mask_base = None if mask_original is None else mask_original[:h_min, :w_min]
    validation_mask_base = None if validation_mask_original is None else validation_mask_original[:h_min, :w_min]

    # ---------------------------------------------------------
    # Zarr-Arrays für alle Crops anlegen
    # ---------------------------------------------------------

    crop_infos = []

    for crop in crops:
        if crop is None:
            segment_name = folder.name
            volume_shape = (d, h_min, w_min)

            label = label_base
            mask = mask_base
            validation_mask = validation_mask_base
        else:
            y_start, y_end, x_start, x_end = crop

            if not (0 <= y_start < y_end <= h_min):
                raise ValueError(f"Invalid Y crop {y_start}:{y_end} for segment {folder.name} with height {h_min}")

            if not (0 <= x_start < x_end <= w_min):
                raise ValueError(f"Invalid X crop {x_start}:{x_end} for segment {folder.name} with width {w_min}")

            segment_name = f"{folder.name}_crop_y_{y_start}_{y_end}_x_{x_start}_{x_end}"

            volume_shape = (d, y_end - y_start, x_end - x_start)

            label = None if label_base is None else label_base[y_start:y_end, x_start:x_end]
            mask = None if mask_base is None else mask_base[y_start:y_end, x_start:x_end]
            validation_mask = None if validation_mask_base is None else validation_mask_base[y_start:y_end, x_start:x_end]

        output_path = output_root / folder.parent.parent.name / folder.parent.name / segment_name

        completion_marker = output_path / ".conversion_complete"
        if completion_marker.is_file():
            print(f"Skipping completed: {segment_name}")
            continue

        output_path.mkdir(parents=True, exist_ok=True)

        z = zarr.open(
            output_path / f"{segment_name}.zarr",
            mode="w"
        )

        z.create_array(
            "volume",
            shape=volume_shape,
            chunks=chunks,
            dtype=first_slice.dtype,
        )

        if label is not None:
            z.create_array(
                "label",
                shape=label.shape,
                chunks=chunks[1:],
                dtype=label.dtype,
            )
            z["label"][:] = label

        if mask is not None:
            z.create_array(
                "mask",
                shape=mask.shape,
                chunks=chunks[1:],
                dtype=mask.dtype,
            )
            z["mask"][:] = mask

        if validation_mask is not None:
            z.create_array(
                "validation_mask",
                shape=validation_mask.shape,
                chunks=chunks[1:],
                dtype=validation_mask.dtype,
            )
            z["validation_mask"][:] = validation_mask

        crop_infos.append({
            "crop": crop,
            "z": z,
            "segment_name": segment_name,
            "completion_marker": completion_marker,
        })

        meta = {
            "volume_original_shape": [d, h, w],
            "label_original_shape": None if label_original is None else label_original.shape,
            "mask_original_shape": None if mask_original is None else mask_original.shape,
            "validation_mask_original_shape": None if validation_mask_original is None else validation_mask_original.shape,
            "volume_shape": volume_shape,
            "label_shape": None if label is None else label.shape,
            "mask_shape": None if mask is None else mask.shape,
            "validation_mask_shape": None if validation_mask is None else validation_mask.shape,
            "chunks": chunks,
            "crop": None if crop is None else {
                "y_start": y_start,
                "y_end": y_end,
                "x_start": x_start,
                "x_end": x_end,
            }
        }
    
        with open(output_path / "metadata.json", "w") as f:
            json.dump(meta, f, indent=4)

    # ---------------------------------------------------------
    # TIFFs Ebene für Ebene laden und in ALLE Crops schreiben
    # ---------------------------------------------------------

    if not crop_infos:
        return f"Skipping completed folder: {folder.name}"

    for i, file in enumerate(volume_files):
        img = tifffile.imread(file)

        for info in crop_infos:
            crop = info["crop"]
            volume_ds = info["z"]["volume"]

            if crop is None:
                volume_ds[i] = img[:h_min, :w_min]
            else:
                y_start, y_end, x_start, x_end = crop

                volume_ds[i] = img[y_start:y_end, x_start:x_end]

    # Only complete stores may be skipped on the next invocation.
    for info in crop_infos:
        info["completion_marker"].touch()

    return f"Finished processing folder: {folder.name}"


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

    crops = _parse_crops(args.crops)

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = []
        
        # for folder in folders:
        #     futures.append(executor.submit(
        #         _process_folder,
        #         folder,
        #         args.output,
        #         args.chunks
        #     ))

        # for folder in folders:
        #     segment_crops = crops.get(folder.name)

        #     if segment_crops is None:
        #         segment_crops = [None]

        #     for crop in segment_crops:
        #         futures.append(
        #             executor.submit(
        #                 _process_folder,
        #                 folder,
        #                 args.output,
        #                 args.chunks,
        #                 crop
        #             )
        #         )

        for folder in folders:
            segment_crops = crops.get(folder.name)

            if segment_crops is None:
                segment_crops = [None]

            futures.append(
                executor.submit(
                    _process_folder,
                    folder,
                    args.output,
                    args.chunks,
                    segment_crops
                )
            )

        for future in as_completed(futures):
            print(future.result())


if __name__ == "__main__":
    main(_get_arguments())
