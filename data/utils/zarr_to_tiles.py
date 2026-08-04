import argparse
from pathlib import Path

import numpy as np
import zarr
from tqdm import tqdm


"""
Generate 2D tile indices from Zarr datasets.

The script processes individual sample folders containing previously generated
Zarr archives. For each segment, the Zarr archive is opened and the spatial
dimensions of the image volume are used to generate tile coordinates.
Optional annotation arrays (label and mask) are used to calculate statistics
for each tile.

For each generated tile, the script stores:
    - segment: name of the processed sample or segment
    - y: vertical tile position in the original image
    - x: horizontal tile position in the original image
    - positive_fraction: fraction of non-zero label pixels within the tile
    - mask_fraction: fraction of non-zero mask pixels within the tile

Tiles are not extracted or stored as image patches. Instead, the script creates
an index containing spatial coordinates and annotation statistics. This allows
efficient sampling of regions from large Zarr datasets during later processing
steps.

The generated tile indices are stored as NumPy binary files (.npy) and exported
as CSV files for inspection and analysis. Multiple tile sizes and strides can
be generated in a single run.

Input:
    A dataset directory containing one subdirectory per segment. Each segment
    directory must contain a corresponding Zarr archive:

        <segment>/<segment>.zarr

The Zarr archive should contain:
        - volume: image data
        - label: optional annotation mask
        - mask: optional validity mask

Output:
    For each processed segment, a "tiles" directory is created containing:

        - tile_size_<size>_stride_<stride>.npy
            Binary tile index containing tile coordinates and metadata.

        - tile_size_<size>_stride_<stride>.csv
            Human-readable version of the tile index.

Usage:
    python zarr_to_tiles.py <input_dir>

Example:
    python zarr_to_tiles.py ./zarr_data

Optional arguments:
    --tile-sizes SIZE [SIZE ...]
        Define one or more square tile sizes to generate.
        Default:
            64 128 192 256 512 1024

        Example:
            --tile-sizes 64 128 256

    --strides STRIDE [STRIDE ...]
        Define the stride for each tile size.
        If omitted, the stride is half the corresponding tile size,
        resulting in overlapping tiles.

        Example:
            --tile-sizes 64 128 \
            --strides 32 64
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
        default=[64, 128, 192, 256, 512, 1024],
        help=(
            "Square tile sizes. "
            "Example: 64 128 192 256 512 1024"
        ),
    )

    parser.add_argument(
        "--strides",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Stride for each tile size. "
            "If omitted, stride equals half the tile size."
        ),
    )

    args = parser.parse_args()

    return args

def main(args):
    tile_sizes = args.tile_sizes

    if args.strides is None:
        strides = [size // 2 for size in tile_sizes]
    else:
        strides = args.strides

    if len(tile_sizes) != len(strides):
        raise ValueError("tile_sizes and strides must have same length")

    for subfolder in sorted(args.input.iterdir()):
        for segment_folder in sorted(subfolder.iterdir()):
            if not segment_folder.is_dir():
                continue

            segment_name = segment_folder.name

            print(f"Processing segment: {segment_name}")

            z = zarr.open(
                segment_folder / f"{segment_name}.zarr",
                mode="r",
            )

            # if "label" not in z or "mask" not in z:
            #     print(f"Skipping segment {segment_name} due to missing label or mask")
            #     continue

            volume = z["volume"]
            label = z["label"] if "label" in z else None
            mask = z["mask"] if "mask" in z else None

            if label is None:
                print(f"Segment {segment_name}: no label found")
            if mask is None:
                print(f"Segment {segment_name}: no mask found")

            output_path = segment_folder / "tiles"
            # if output_path.exists():
            #     print(f"Tiles already exist for segment: {segment_name}. Skipping.")
            #     continue
            output_path.mkdir(parents=True, exist_ok=True)

            # h, w = label.shape
            h, w = volume.shape[-2:]

            for size, stride in zip(tile_sizes, strides):
                npy_file = segment_folder / "tiles" / f"tile_size_{size}_stride_{stride}.npy"
                csv_file = segment_folder / "tiles" / f"tile_size_{size}_stride_{stride}.csv"
            
                if npy_file.exists():
                    print(f"Tile (size: {size}, stride: {stride}) already exists for segment: {segment_name}. Skipping.")
                    continue
                
                l = []

                with tqdm(total=((h - size) // stride + 1) * ((w - size) // stride + 1), desc=f"Tile size {size}, stride {stride}") as pbar:
                    for y in range(0, h - size + 1, stride):
                        for x in range(0, w - size + 1, stride):
                            # label_tile = label[y:y+size, x:x+size]
                            # mask_tile = mask[y:y+size, x:x+size]

                            if label is not None:
                                label_tile = label[y:y+size, x:x+size]
                                positive_fraction = float(label_tile.mean())
                            else:
                                positive_fraction = np.nan

                            if mask is not None:
                                mask_tile = mask[y:y+size, x:x+size]
                                mask_fraction = float(mask_tile.mean())
                            else:
                                mask_fraction = np.nan

                            l.append(
                                (
                                    segment_name,
                                    y,
                                    x,
                                    positive_fraction,
                                    mask_fraction,
                                )
                            )

                            pbar.update(1)

                l = np.asarray(l, dtype=TILE_DTYPE)
                np.save(npy_file, l)

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
