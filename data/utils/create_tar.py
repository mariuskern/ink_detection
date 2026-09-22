import argparse
from pathlib import Path
import tarfile
from concurrent.futures import ThreadPoolExecutor, as_completed
import shutil
import subprocess


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


def create_tar(segment_folder, output_tar, tar_path, pigz_path):
    # tar_path = shutil.which("tar")
    # pigz_path = shutil.which("pigz")

    if tar_path:
        if pigz_path:
            print("Creating archive using tar + pigz")
            cmd = [
                tar_path,
                "-I", f"{pigz_path} -1",
                "-cf",
                str(output_tar),
                "-C",
                str(segment_folder.parent),
                segment_folder.name,
            ]
        else:
            print("Creating archive using tar + gzip")
            cmd = [
                tar_path,
                "-czf",
                str(output_tar),
                "-C",
                str(segment_folder.parent),
                segment_folder.name,
            ]

        subprocess.run(cmd, check=True)

    else:
        print("tar not found, using Python tarfile")
        with tarfile.open(output_tar, "w:gz") as tar:
            tar.add(
                segment_folder,
                arcname=segment_folder.name,
            )


def _process_folder(output_root, scroll, category, segment_folder, tar_path, pigz_path):
    output_subdir = output_root / scroll.name / category.name
    output_subdir.mkdir(parents=True, exist_ok=True)

    output_tar = output_subdir / f"{segment_folder.name}.tar.gz"

    # print(f"Creating archive: {output_tar}")

    # with tarfile.open(output_tar, "w:gz") as tar:
    #     tar.add(
    #         segment_folder,
    #         arcname=segment_folder.name,
    #     )

    create_tar(segment_folder, output_tar, tar_path, pigz_path)

    return f"Created archive: {output_tar}"


def main(args):
    if not args.input_dir.is_dir():
        raise ValueError(f"Input directory does not exist: {args.input_dir}")

    if args.output_dir.exists():
        raise ValueError(f"Output directory already exists: {args.output_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    tar_path = shutil.which("tar")
    pigz_path = shutil.which("pigz")

    if tar_path:
        if pigz_path:
            print("Using Linux tar + pigz")
        else:
            print("Using Linux tar + gzip")
    else:
        print("tar not found, using Python tarfile")

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = []

        for scroll in args.input_dir.iterdir():
            for category in scroll.iterdir():
                for segment_folder in category.iterdir():
                    if not segment_folder.is_dir():
                        continue
                    
                    futures.append(executor.submit(
                        _process_folder,
                        args.output_dir,
                        scroll,
                        category,
                        segment_folder,
                        tar_path,
                        pigz_path,
                    ))
            
        for future in as_completed(futures):
            print(future.result())

    print("Finished creating all archives.")


if __name__ == "__main__":
    main(_get_arguments())
