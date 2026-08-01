import argparse
from pathlib import Path
import tarfile


def _get_arguments():
    parser = argparse.ArgumentParser(
        description="Create a tar archive containing all subfolders of a directory."
    )

    parser.add_argument(
        "input_dir",
        type=Path,
        help="Directory containing the subfolders to archive.",
    )

    parser.add_argument(
        "output_dir",
        type=Path,
        help="Directory where the tar archives will be stored.",
    )

    return parser.parse_args()


def main(args):
    if not args.input_dir.is_dir():
        raise ValueError(f"Input directory does not exist: {args.input_dir}")

    if args.output_dir.exists():
        raise ValueError(f"Output directory already exists: {args.output_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    for subfolder in args.input_dir.iterdir():
        if not subfolder.is_dir():
            continue

        output_tar = args.output_dir / f"{subfolder.name}.tar"

        print(f"Creating archive: {output_tar}")

        with tarfile.open(output_tar, "w") as tar:
            tar.add(
                subfolder,
                arcname=subfolder.name,
            )

    print("Finished creating all archives.")


if __name__ == "__main__":
    main(_get_arguments())
