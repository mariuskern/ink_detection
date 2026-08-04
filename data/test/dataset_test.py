import os
import sys

import hydra
from omegaconf import DictConfig, OmegaConf

from dotenv import load_dotenv
load_dotenv()

sys.path.append(os.getenv("CODE_DIR"))

from data import InkDataset, create_balanced_sampler, create_dataloader


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg : DictConfig) -> None:
    dataset = InkDataset(
        cfg.dataset.dir,
        segments=cfg.dataset.train,
        tile_size=cfg.dataset.tile_size,
        stride=cfg.dataset.stride,
        n_layers=cfg.dataset.n_layers,
        center=cfg.dataset.center,
        depth=cfg.dataset.depth,
        norm=cfg.dataset.norm,
    )

    print(len(dataset))
    tile, label, segment, x, y = dataset[2]
    print(f"Tile shape: {tile.shape}, Label shape: {label.shape}, Segment: {segment}, x: {x}, y: {y}")

    sampler = create_balanced_sampler(dataset, cfg.sampler.alpha)

    loader = create_dataloader(
        dataset,
        sampler=sampler,
        batch_size=cfg.dataloader.batch_size,
        shuffle=cfg.dataloader.shuffle,
        num_workers=cfg.dataloader.num_workers,
        pin_memory=cfg.dataloader.pin_memory
    )

    # tile, label, segment, x, y = next(iter(loader))
    # print(tile.shape)

    for i, batch in list(enumerate(loader))[:25]:
        tile, label, segment, x, y = batch
        print(tile.shape)

if __name__ == "__main__":
    main()
