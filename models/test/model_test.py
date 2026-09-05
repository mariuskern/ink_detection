import os
import sys
import argparse
from pathlib import Path

import torch

from hydra import compose, initialize
from omegaconf import DictConfig

from dotenv import load_dotenv
load_dotenv()

sys.path.append(os.getenv("CODE_DIR"))

from models import build_model
from data import build_train_dataset, build_sampler, build_dataloader


def _get_arguments():
    parser = argparse.ArgumentParser(
        description="Convert all downloaded Zarr segments into TIFF stacks while preserving the directory structure."
    )

    parser.add_argument(
        "--checkpoint_path",
        type=Path,
        default=None,
        help="Path to the checkpoint file.",
    )

    return parser.parse_args()


def main(args) -> None:
    with initialize(config_path="../../conf/", version_base=None):
        cfg = compose(config_name="debug")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(cfg).to(device)
    print("Model created")

    if args.checkpoint_path is not None and args.checkpoint_path.exists():
        model.load_vesuvius_checkpoint(args.checkpoint_path, map_location="cuda")
        print("Checkpoint loaded")

    dataset = build_train_dataset(cfg)
    sampler = build_sampler(cfg, dataset)
    loader = build_dataloader(cfg, dataset, sampler)

    print("Loader created")

    for i, batch in list(enumerate(loader))[:25]:
        # tile, label, mask, is_pseudo, segment, x, y, _ = batch
        tile = batch["tile"]
        tile = tile.to(device)
        out = model(tile)
        print(out.shape)
        print(i)


if __name__ == "__main__":
    main(_get_arguments())
