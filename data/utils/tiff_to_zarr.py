import argparse
from pathlib import Path
from PIL import Image
import json
# from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import tifffile
import zarr
from tqdm import tqdm

Image.MAX_IMAGE_PIXELS = None


"""
Conversion pipeline for transforming volumetric TIFF image stacks into Zarr datasets.

The script processes individual sample folders containing 3D TIFF image stacks
and optional annotation images (labels and masks). Each TIFF stack is loaded as
a NumPy volume, spatial dimensions are aligned between image and annotation data,
and the resulting arrays are stored as chunked Zarr datasets.

The generated Zarr archives contain separate arrays for:
    - volume: 3D image data (Z, Y, X)
    - label: 2D binary annotation map (Y, X), if available
    - mask: 2D binary mask (Y, X), if available

Additionally, metadata describing original and processed array shapes as well
as chunk sizes is stored as a JSON file. Chunked Zarr storage enables efficient
partial loading of large volumetric datasets, making the format suitable for
out-of-core processing and deep learning workflows on large microscopy volumes.

Usage:
    python tiff_to_zarr.py <input_dir> <output_dir>

Example:
    python tiff_to_zarr.py ./tiff_data ./zarr_data

Optional arguments:
    --segments SEGMENT [SEGMENT ...]
        Convert only the specified sample folders.
        By default, all subfolders in the input directory are processed.

    --chunks Z Y X
        Define the Zarr chunk size for the volumetric image data.
        The default chunk size is optimized for 3D volume access.

Example with optional arguments:
    python tiff_to_zarr.py ./tiff_data ./zarr_data \
        --segments sample_1 sample_2 \
        --chunks 65 512 512
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


# def _get_volume(folder):
#     layers = folder / "layers"

#     files = sorted(layers.glob("*.tif"))

#     if not files:
#         print(f"No TIFF files found in folder: {folder.name}")
#         return None

#     # with ThreadPoolExecutor() as executor:
#     #     slices = list(executor.map(tifffile.imread, files))

#     # volume = np.stack(slices)
    
#     volume = np.stack([tifffile.imread(f) for f in files])

#     return volume


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


def _load_binary_image(path):
    image = np.array(Image.open(path).convert("L"))
    binary_image = (image > 127).astype(np.uint8)
    return binary_image


# def _crop_images(volume, label=None, mask=None):
#     arrays = [volume]

#     if label is not None:
#         arrays.append(label)
    
#     if mask is not None:
#         arrays.append(mask)

#     h_min = min(arr.shape[-2] for arr in arrays)
#     w_min = min(arr.shape[-1] for arr in arrays)

#     volume = volume[:, :h_min, :w_min]
#     label = None if label is None else label[:h_min, :w_min]
#     mask = None if mask is None else mask[:h_min, :w_min]

#     return volume, label, mask


# def _load_and_crop(args):
#     index, file, h_min, w_min = args

#     img = tifffile.imread(file)
#     img = img[:h_min, :w_min]

#     return index, img


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

        # volume_original = _get_volume(folder)
        volume_files = _get_volume_files(folder)
        label_original = _get_label(folder)
        mask_original = _get_mask(folder)

        # if volume_original is None:
        #     print(f"Skipping folder {folder.name} due to missing image stack.")
        #     continue

        if volume_files is None:
            print(f"Skipping folder {folder.name} due to missing image stack.")
            continue

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

        h_min = min(a.shape[-2] for a in arrays)
        w_min = min(a.shape[-1] for a in arrays)

        volume_shape = (d, h_min, w_min)

        label = None if label_original is None else label_original[:h_min, :w_min]
        mask = None if mask_original is None else mask_original[:h_min, :w_min]

        meta = {
            "volume_original_shape": (d, h, w),
            "label_original_shape": None if label_original is None else label_original.shape,
            "mask_original_shape": None if mask_original is None else mask_original.shape,
            "volume_shape": volume_shape,
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
            shape=volume_shape,
            chunks=args.chunks,
            dtype=first_slice.dtype,
        )

        # z["volume"][:] = volume

        volume_ds = z["volume"]

        for i, file in tqdm(enumerate(volume_files), desc="Loading and saving TIFFs", total=len(volume_files)):
            img = tifffile.imread(file)
            volume_ds[i] = img[:h_min, :w_min]

        # tasks = [
        #     (i, file, h_min, w_min)
        #     for i, file in enumerate(volume_files)
        # ]

        # with ThreadPoolExecutor(max_workers=8) as executor:
        #     futures = [
        #         executor.submit(_load_and_crop, task)
        #         for task in tasks
        #     ]

        #     for future in tqdm(
        #         as_completed(futures),
        #         total=len(futures),
        #         desc="Loading TIFFs"
        #     ):
        #         i, img = future.result()
        #         volume_ds[i] = img


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
    main(_get_arguments())
