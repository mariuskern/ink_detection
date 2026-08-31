import argparse
import shutil
from pathlib import Path

import numpy as np
import tifffile
import zarr
from tqdm import tqdm

"""
Convert all downloaded Zarr segments into TIFF image stacks.

The input directory is recursively searched for .zarr directories.

The output directory keeps exactly the same directory structure as
the input directory.

Example input:

    hf_data/
    ├── 9b/
    │   ├── labeled/
    │   │   └── segment_A/
    │   │       ├── segment_A.zarr/
    │   │       ├── segment_A_inklabels.png
    │   │       ├── segment_A_mask.png
    │   │       ├── meta.json
    │   │       └── meta_download_info.json
    │   │
    │   └── unlabeled/
    │       └── segment_B/
    │           └── segment_B.zarr/


Example output:

    tiff_data/
    ├── 9b/
    │   ├── labeled/
    │   │   └── segment_A/
    │   │       ├── layers/
    │   │       │   ├── 00.tif
    │   │       │   ├── 01.tif
    │   │       │   └── ...
    │   │       ├── segment_A_inklabels.png
    │   │       ├── segment_A_mask.png
    │   │       ├── meta.json
    │   │       └── meta_download_info.json
    │   │
    │   └── unlabeled/
    │       └── segment_B/
    │           └── layers/
    │               ├── 00.tif
    │               ├── 01.tif
    │               └── ...


The .zarr directory itself is NOT copied.

Only the volume inside the Zarr is converted to TIFF.

Usage:

    python zarr_to_tiff.py <input_dir> <output_dir>

Example:

    python zarr_to_tiff.py \
        "D:/Dokumente/Studium/Masterarbeit/hf_data" \
        "D:/Dokumente/Studium/Masterarbeit/tiff_data"

Optional arguments:

    --prefix PREFIX
        Prefix for generated TIFF files.

    --digits DIGITS
        Number of digits used for zero-padding.
        Default: 3

    --overwrite
        Overwrite existing TIFF files.
"""


def _get_arguments():
    parser = argparse.ArgumentParser(
        description="Convert all downloaded Zarr segments into TIFF stacks while preserving the directory structure."
    )

    parser.add_argument(
        "input_dir",
        type=Path,
        help="Root directory containing the downloaded segments.",
    )

    parser.add_argument(
        "output_dir",
        type=Path,
        help="Root directory where the converted data will be stored.",
    )

    parser.add_argument(
        "--prefix",
        type=str,
        default="",
        help="Prefix for generated TIFF files (default: none).",
    )

    parser.add_argument(
        "--digits",
        type=int,
        default=2,
        help="Number of digits for zero-padding (default: 3).",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing TIFF files and copied files.",
    )

    return parser.parse_args()


def find_segments(root_dir):
    """
    Find all .zarr directories recursively.

    Returns:
        List of tuples:
            (segment_dir, zarr_path)
    """

    zarr_paths = sorted(path for path in root_dir.rglob("*.zarr") if path.is_dir())

    segments = []

    for zarr_path in zarr_paths:
        segment_dir = zarr_path.parent
        segments.append((segment_dir, zarr_path))

    return segments


def copy_additional_files(input_segment_dir, output_segment_dir, zarr_path, overwrite):
    """
    Copy all files from the input segment directory except
    the .zarr directory.

    This includes:
        - meta.json
        - meta_download_info.json
        - *_inklabels.png
        - *_mask.png
        - any other regular files

    The Zarr directory itself is not copied.
    """

    output_segment_dir.mkdir(parents=True, exist_ok=True)

    copied_files = 0

    for item in input_segment_dir.iterdir():
        if item == zarr_path:
            continue

        if not item.is_file():
            continue

        output_file = output_segment_dir / item.name

        if output_file.exists() and not overwrite:
            continue

        shutil.copy2(item, output_file)

        copied_files += 1

        print(f"Copied: {item.name}")

    return copied_files


def convert_segment(input_segment_dir, output_segment_dir, zarr_path, prefix, digits, overwrite):
    """
    Convert one Zarr segment into TIFF slices and copy
    all additional files to the corresponding output directory.
    """

    segment_name = zarr_path.stem

    print("")
    print("=" * 70)
    print(f"Segment: {segment_name}")
    print(f"Input:   {input_segment_dir}")
    print(f"Output:  {output_segment_dir}")
    print("=" * 70)
    print("")


    # --------------------------------------------------------
    # Create output segment directory
    # --------------------------------------------------------

    output_segment_dir.mkdir(parents=True, exist_ok=True)


    # --------------------------------------------------------
    # Copy additional files
    # --------------------------------------------------------

    print("Copying additional files...")
    copied_files = copy_additional_files(input_segment_dir=input_segment_dir, output_segment_dir=output_segment_dir, zarr_path=zarr_path, overwrite=overwrite)
    print(f"Copied {copied_files} file(s).")
    print("")


    # --------------------------------------------------------
    # Open Zarr
    # --------------------------------------------------------

    print("Opening Zarr...")
    z = zarr.open(zarr_path, mode="r")
    print("Zarr structure:")
    print(z.tree())
    print("")


    # --------------------------------------------------------
    # Find volume
    # --------------------------------------------------------

    if "0" not in z:
        available = list(z.keys())

        raise ValueError(
            f"No '0' array found in:\n"
            f"{zarr_path}\n\n"
            f"Available arrays/groups:\n"
            f"{available}"
        )

    volume = z["0"]


    # --------------------------------------------------------
    # Check dimensions
    # --------------------------------------------------------

    if volume.ndim != 3:
        raise ValueError(
            f"Expected a 3D volume with shape (Z, Y, X), but got:\n"
            f"{volume.shape}"
        )

    d, h, w = volume.shape

    print(f"Volume shape:     {volume.shape}")
    print(f"Dtype:            {volume.dtype}")
    print(f"Chunks:           {volume.chunks}")
    print(f"Number of slices: {d}")
    print("")


    # --------------------------------------------------------
    # Create layers directory
    # --------------------------------------------------------

    layers_dir = output_segment_dir / "layers"
    layers_dir.mkdir(parents=True, exist_ok=True)
    print(f"Layers directory:\n" f"  {layers_dir}")
    print("")
    

    # --------------------------------------------------------
    # Convert slices
    # --------------------------------------------------------

    for i in tqdm(range(d), desc=f"Converting {segment_name}", unit="slice"):
        output_file = layers_dir / f"{prefix}{i:0{digits}d}.tif"

        if output_file.exists() and not overwrite:
            continue

        image = np.asarray(volume[i])

        tifffile.imwrite(output_file, image)

    print("")
    print(f"Finished: {segment_name}")


def main(args):

    # ========================================================
    # Check input directory
    # ========================================================

    input_dir = args.input_dir
    output_dir = args.output_dir

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found:\n" f"{input_dir}")
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input path is not a directory:\n" f"{input_dir}")

    # ========================================================
    # Prevent output directory inside input directory
    # ========================================================

    input_dir_resolved = input_dir.resolve()
    output_dir_resolved = output_dir.resolve()

    if output_dir_resolved == input_dir_resolved or input_dir_resolved in output_dir_resolved.parents:
        raise ValueError(
            "The output directory must not be inside the input directory.\n\n"
            f"Input:  {input_dir_resolved}\n"
            f"Output: {output_dir_resolved}"
        )

    # ========================================================
    # Find segments
    # ========================================================

    print("")
    print("=" * 70)
    print("Searching for Zarr segments")
    print("=" * 70)
    print("")
    print(f"Input directory:\n" f"  {input_dir}")
    print(f"Output directory:\n" f"  {output_dir}")
    print("")

    segments = find_segments(input_dir)

    if not segments:
        print("No .zarr directories found.")
        return

    print(f"Found {len(segments)} Zarr segment(s):")
    print("")

    for segment_dir, zarr_path in segments:
        relative_segment_dir = segment_dir.relative_to(input_dir)
        print(f"  {relative_segment_dir}")

    print("")

    # ========================================================
    # Convert all segments
    # ========================================================

    successful = 0
    failed = 0

    for index, (input_segment_dir, zarr_path) in enumerate(segments, start=1):
        relative_segment_dir = input_segment_dir.relative_to(input_dir)
        output_segment_dir = output_dir / relative_segment_dir

        print("")
        print(f"[{index}/{len(segments)}]")

        try:
            convert_segment(
                input_segment_dir=input_segment_dir,
                output_segment_dir=output_segment_dir,
                zarr_path=zarr_path,
                prefix=args.prefix,
                digits=args.digits,
                overwrite=args.overwrite,
            )

            successful += 1
        except Exception as e:
            failed += 1

            print("")
            print(f"ERROR processing " f"{input_segment_dir.name}:")

            print(f"  {e}")

    # ========================================================
    # Summary
    # ========================================================

    print("")
    print("=" * 70)
    print("Conversion completed")
    print("=" * 70)
    print("")
    print(f"Successful: {successful}")
    print(f"Failed:     {failed}")
    print(f"Total:      {len(segments)}")
    print("")


if __name__ == "__main__":
    main(_get_arguments())


# import argparse
# from pathlib import Path

# import numpy as np
# import tifffile
# import zarr
# from tqdm import tqdm


# """
# Conversion pipeline for transforming a Zarr volume into a TIFF image stack.

# The input is a single Zarr store containing a 3D volume with shape (Z, Y, X).
# The volume is read slice-by-slice, so the complete volume does not need to be
# loaded into memory at once.

# Expected Zarr structure:

#     segment.zarr/
#     └── volume
#         shape = (Z, Y, X)

# The output directory will contain one TIFF file per Z-slice:

#     output/
#     ├── 000.tif
#     ├── 001.tif
#     ├── 002.tif
#     └── ...

# Usage:
#     python zarr_to_tiff.py <input_zarr> <output_dir>

# Example:
#     python zarr_to_tiff.py segment.zarr ./layers

# Optional arguments:
#     --prefix PREFIX
#         Prefix for the generated TIFF files.

#     --digits DIGITS
#         Number of digits used for zero-padding the slice numbers.
#         Default: 3

#     --overwrite
#         Overwrite existing TIFF files.
# """


# def _get_arguments():
#     parser = argparse.ArgumentParser(
#         description="Convert a Zarr volume into a TIFF image stack"
#     )

#     parser.add_argument(
#         "input",
#         type=Path,
#         help="Input Zarr store containing a 'volume' array",
#     )

#     parser.add_argument(
#         "output",
#         type=Path,
#         help="Output directory for TIFF slices",
#     )

#     parser.add_argument(
#         "--prefix",
#         type=str,
#         default="",
#         help="Prefix for generated TIFF files (default: none)",
#     )

#     parser.add_argument(
#         "--digits",
#         type=int,
#         default=3,
#         help="Number of digits for zero-padding (default: 3)",
#     )

#     parser.add_argument(
#         "--overwrite",
#         action="store_true",
#         help="Overwrite existing TIFF files",
#     )

#     return parser.parse_args()


# def main(args):
#     # Check input
#     if not args.input.exists():
#         raise FileNotFoundError(
#             f"Zarr store not found: {args.input}"
#         )

#     # Open Zarr
#     z = zarr.open(args.input, mode="r")

#     print(z.tree())

#     for name in z.keys():
#         item = z[name]
#         print(
#             f"{name}: "
#             f"type={type(item)}, "
#             f"shape={getattr(item, 'shape', None)}, "
#             f"dtype={getattr(item, 'dtype', None)}"
#         )

#     print("")

#     if "0" not in z:
#         available = list(z.keys())
#         raise ValueError(
#             f"No '0' array found in Zarr store: {args.input}\n"
#             f"Available groups/arrays: {available}"
#         )

#     volume = z["0"]

#     if volume.ndim != 3:
#         raise ValueError(
#             f"Expected a 3D volume with shape (Z, Y, X), "
#             f"but got shape {volume.shape}"
#         )

#     d, h, w = volume.shape

#     print(f"Input Zarr: {args.input}")
#     print(f"Volume shape: {volume.shape}")
#     print(f"Dtype: {volume.dtype}")
#     print(f"Chunks: {volume.chunks}")
#     print(f"Number of slices: {d}")
#     print("")

#     # Create output directory
#     args.output.mkdir(parents=True, exist_ok=True)

#     # Save slices
#     for i in tqdm(
#         range(d),
#         desc="Converting Zarr to TIFF"
#     ):
#         output_file = args.output / (
#             f"{args.prefix}{i:0{args.digits}d}.tif"
#         )

#         if output_file.exists() and not args.overwrite:
#             continue

#         # Only load one Z-slice into memory
#         image = np.asarray(volume[i])

#         tifffile.imwrite(
#             output_file,
#             image,
#         )

#     print("")
#     print(f"Saved TIFF stack to: {args.output}")


# if __name__ == "__main__":
#     main(_get_arguments())
