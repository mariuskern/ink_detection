from .dataset import InkDataset
from .sampler import create_balanced_sampler
from .loader import create_dataloader


def build_dataloader(cfg):
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
    dataset_val = InkDataset(
        cfg.dataset.dir,
        segments=cfg.dataset.val,
        tile_size=cfg.dataset.tile_size,
        stride=cfg.dataset.stride,
        n_layers=cfg.dataset.n_layers,
        center=cfg.dataset.center,
        depth=cfg.dataset.depth,
        norm=cfg.dataset.norm,
    )

    sampler = create_balanced_sampler(dataset, cfg.sampler.num_samples, cfg.sampler.alpha)

    loader = create_dataloader(
        dataset,
        sampler=sampler,
        batch_size=cfg.dataloader.batch_size,
        shuffle=cfg.dataloader.shuffle,
        num_workers=cfg.dataloader.num_workers,
        pin_memory=cfg.dataloader.pin_memory
    )

    loader_val = create_dataloader(
        dataset,
        sampler=None,
        batch_size=cfg.dataloader.batch_size,
        shuffle=False,
        num_workers=cfg.dataloader.num_workers,
        pin_memory=cfg.dataloader.pin_memory
    )

    return loader, loader_val