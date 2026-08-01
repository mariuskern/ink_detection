import argparse
from pathlib import Path

import numpy as np
import zarr


"""
Pipeline for generating 2D tile indices from Zarr datasets.

The script processes individual sample folders containing previously generated
Zarr archives. For each segment, the corresponding label and mask arrays are
loaded and divided into square tiles of configurable size and stride. For every
tile, spatial coordinates and summary statistics describing the annotation
content are calculated.

The generated tile indices contain the following information:
    - segment: name of the processed sample or segment
    - y: vertical tile position in the original image
    - x: horizontal tile position in the original image
    - positive_fraction: fraction of positive label pixels within the tile
    - mask_fraction: fraction of valid mask pixels within the tile

The tile indices are stored as NumPy binary files (.npy) for efficient loading
during downstream processing and additionally exported as CSV files for
inspection and analysis. Multiple tile sizes and strides can be generated in a
single run, enabling multi-scale analysis of large microscopy datasets.

Usage:
    python create_tile_index.py <input_dir>

Example:
    python create_tile_index.py ./zarr_data

Optional arguments:
    --tile-sizes SIZE [SIZE ...]
        Define one or more square tile sizes to generate.
        Default: 64

        Example:
            --tile-sizes 16 32 64 128 256

    --strides STRIDE [STRIDE ...]
        Define the stride for each tile size.
        If omitted, the stride is equal to the corresponding tile size,
        resulting in non-overlapping tiles.

        Example:
            --tile-sizes 64 128 \
            --strides 32 64

Output:
    For each processed segment, a "tiles" directory is created containing:
        - tile_size_<size>_stride_<stride>.npy
            Binary tile index containing all tile metadata.

        - tile_size_<size>_stride_<stride>.csv
            Human-readable representation of the tile index.

The generated tile indices enable efficient spatial sampling, visualization,
and selection of image regions for downstream analysis and machine learning
workflows without repeatedly accessing the full-resolution Zarr datasets.
"""


TILE_DTYPE = np.dtype(
    [
        ("segment", "U128"),
        ("y", np.int32),
        ("x", np.int32),
        ("positive_fraction", np.float32),
        ("mask_fraction", np.float32),
    ]
)


def _get_arguments():
    parser = argparse.ArgumentParser(
        description="Create 2D tile indices from Zarr archives."
    )

    parser.add_argument(
        "input",
        type=Path,
        help="Dataset directory containing segment folders.",
    )

    parser.add_argument(
        "--tile-sizes",
        type=int,
        nargs="+",
        default=[64],
        help=(
            "Square tile sizes. "
            "Example: 16 32 64 128 256 512 1024"
        ),
    )

    parser.add_argument(
        "--strides",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Stride for each tile size. "
            "If omitted, stride equals tile size."
        ),
    )

    args = parser.parse_args()

    return args

def main(args):
    tile_sizes = args.tile_sizes

    if args.strides is None:
        strides = tile_sizes
    else:
        strides = args.strides

    if len(tile_sizes) != len(strides):
        raise ValueError("tile_sizes and strides must have same length")

    for segment_folder in sorted(args.input.iterdir()):
        if not segment_folder.is_dir():
            continue

        segment_name = segment_folder.name

        print(f"Processing segment: {segment_name}")

        z = zarr.open(
            segment_folder / f"{segment_name}.zarr",
            mode="r",
        )

        if "label" not in z or "mask" not in z:
            print(f"Skipping segment {segment_name} due to missing label or mask")
            continue

        label = z["label"]
        mask = z["mask"]

        output_path = segment_folder / "tiles"
        if output_path.exists():
            print(f"Tiles already exist for segment: {segment_name}. Skipping.")
            continue
        output_path.mkdir(parents=True)

        h, w = label.shape

        for size, stride in zip(tile_sizes, strides):
            l = []

            for y in range(0, h - size + 1, stride):
                for x in range(0, w - size + 1, stride):
                    label_tile = label[y:y+size, x:x+size]
                    mask_tile = mask[y:y+size, x:x+size]

                    l.append(
                        (
                            segment_name,
                            y,
                            x,
                            float(label_tile.mean()),
                            float(mask_tile.mean()),
                        )
                    )

            l = np.asarray(l, dtype=TILE_DTYPE)

            npy_file = segment_folder / "tiles" / f"tile_size_{size}_stride_{stride}.npy"
            np.save(npy_file, l)

            csv_file = segment_folder / "tiles" / f"tile_size_{size}_stride_{stride}.csv"
            np.savetxt(
                csv_file,
                l,
                delimiter=",",
                header=",".join(l.dtype.names),
                comments="",
                fmt=[
                    "%s",
                    "%d",
                    "%d",
                    "%.6f",
                    "%.6f",
                ],
            )

            print(f"Saved {len(l)} tiles")

        print("")


if __name__ == "__main__":
    main(_get_arguments())
