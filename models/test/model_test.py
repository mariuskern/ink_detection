import os
import sys

import torch

import hydra
from omegaconf import DictConfig, OmegaConf

from dotenv import load_dotenv
load_dotenv()

sys.path.append(os.getenv("CODE_DIR"))

from models import build_model
from data import build_dataloader, build_augmentation


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg : DictConfig) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(cfg).to(device)

    print("Model created")

    # loader, _ = build_dataloader(cfg)
    loader, _ = build_dataloader(cfg, aug=cfg.train.student.aug)

    print("Loader created")

    for i, batch in list(enumerate(loader))[:25]:
        tile, label, mask, is_pseudo, segment, x, y, _ = batch
        tile = tile.to(device)
        out = model(tile)
        print(out.shape)
        print(i)


if __name__ == "__main__":
    main()
