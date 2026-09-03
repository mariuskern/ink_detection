import os
import sys

import hydra
from omegaconf import DictConfig, OmegaConf

from dotenv import load_dotenv
load_dotenv()

sys.path.append(os.getenv("CODE_DIR"))

from data import build_train_dataset, build_sampler, build_dataloader


@hydra.main(version_base=None, config_path="../../conf", config_name="debug")
def main(cfg : DictConfig) -> None:
    # dataset = InkDataset(
    #     cfg.dataset.dir,
    #     segments=cfg.dataset.train,
    #     tile_size=cfg.dataset.tile_size,
    #     stride=cfg.dataset.stride,
    #     n_layers=cfg.dataset.n_layers,
    #     center=cfg.dataset.center,
    #     depth=cfg.dataset.depth,
    #     norm=cfg.dataset.norm,
    # )

    dataset = build_train_dataset(cfg)

    print(len(dataset))
    batch = dataset[0]
    # print(f"Tile shape: {tile.shape}, Label shape: {label.shape}, Segment: {segment}, x: {x}, y: {y}")
    print(f"Tile shape: {batch["tile"].shape}, Label shape: {batch["label"].shape}, Segment: {batch["segment"]}")

    sampler = build_sampler(cfg, dataset)
    loader = build_dataloader(cfg, dataset, sampler)

    # tile, label, segment, x, y = next(iter(loader))
    # print(tile.shape)

    for i, batch in list(enumerate(loader))[:25]:
        # tile, label, segment, x, y = batch
        tile = batch["tile"]
        print(tile.shape)

if __name__ == "__main__":
    main()
