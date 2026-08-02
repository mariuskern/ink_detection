import os
import sys

import hydra
from omegaconf import DictConfig, OmegaConf

from dotenv import load_dotenv
load_dotenv()

sys.path.append(os.getenv("CODE"))

from data import InkDataset


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg : DictConfig) -> None:
    dataset = InkDataset(cfg.dataset.dir, segments=cfg.dataset.train, tile_size=cfg.dataset.tile_size)
    print(len(dataset))
    tile, label, segment, x, y = dataset[2]
    print(f"Tile shape: {tile.shape}, Label shape: {label.shape}, Segment: {segment}, x: {x}, y: {y}")

if __name__ == "__main__":
    main()
