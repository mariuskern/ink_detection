import os
import shutil
from pathlib import Path
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed

from hydra import compose, initialize

from dotenv import load_dotenv
load_dotenv()
for key in ["DATA_DIR", "CODE_DIR", "OUTPUT_DIR"]:
    if key in os.environ:
        os.environ[key] = os.path.expandvars(os.environ[key])


def copy_and_extract_archive(src_file, tar_dest_folder, zarr_dest_folder):
    tar_dest_folder.mkdir(parents=True, exist_ok=True)
    zarr_dest_folder.mkdir(parents=True, exist_ok=True)

    shutil.copy2(src_file, tar_dest_folder)

    subprocess.run(
        [
            "tar",
            "-xzf",
            str(tar_dest_folder / src_file.name),
            "-C",
            str(zarr_dest_folder),
        ],
        check=True,
    )

    return f"Copied and extracted {src_file.name}"


# @hydra.main(version_base=None, config_path="../../conf/", config_name="-provide_as_argument-")
def main():
    with initialize(config_path="../../conf/", version_base=None):
        cfg = compose(config_name="config")
    
    stage_dir = Path(cfg.path.stage_dir) # data_tar
    data_dir = Path(cfg.path.data_dir) # data_zarr

    # segments = set(map(str, cfg.dataset.train)) | set(map(str, cfg.dataset.val))
    segments = set(map(str, cfg.train.labeled)) | set(map(str, cfg.train.val)) | set(map(str, getattr(cfg.train, "unlabeled", []) or []))

    tasks = []
    for category in stage_dir.iterdir():  # labeled/, unlabeled/, ...
        if not category.is_dir():
            continue

        tar_dest_folder = data_dir.parent / stage_dir.name / category.name
        zarr_dest_folder = data_dir / category.name
        # target_category.mkdir(parents=True, exist_ok=True)

        for archive in category.glob("*.tar.gz"):
            segment_name = archive.name.removesuffix(".tar.gz")

            if segment_name not in segments:
                continue

            tasks.append((archive, tar_dest_folder, zarr_dest_folder))

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(copy_and_extract_archive, src_file, tar_dest_folder, zarr_dest_folder) for src_file, tar_dest_folder, zarr_dest_folder in tasks]

        for future in as_completed(futures):
            print(future.result())


if __name__ == "__main__":
    main()
