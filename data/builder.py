import torch
from torchvision.transforms import v2

from .dataset import InkDataset, CombinedInkDataset
from .sampler import create_balanced_sampler
from .loader import create_dataloader


class AddGaussianNoise(torch.nn.Module):
    def __init__(self, mean=0., sigma=0.05):
        super().__init__()

        self.mean = mean
        self.sigma = sigma

    def forward(self, image, *masks):
        noise = torch.randn_like(image)
        image = image + noise * self.sigma + self.mean
        image = torch.clamp(image, 0, 1)
        return (image, *masks)


def build_augmentation(aug):
    transforms = []

    for t in aug.transforms:
        match t.name:
            case "random_horizontal_flip":
                transforms.append(v2.RandomHorizontalFlip(p=t.p))
            case "random_vertical_flip":
                transforms.append(v2.RandomVerticalFlip(p=t.p))
            case "random_rotation":
                transforms.append(v2.RandomRotation(degrees=t.degrees))
            case "random_affine":
                transforms.append(v2.RandomAffine(degrees=t.degrees, translate=t.translate, scale=t.scale, shear=t.shear))
            case "random_resized_crop":
                transforms.append(v2.RandomResizedCrop(size=t.size, scale=t.scale, ratio=t.ratio))
            case "color_jitter":
                transforms.append(v2.ColorJitter(brightness=t.brightness, contrast=t.contrast, saturation=t.saturation, hue=t.hue))
            case "gaussian_blur":
                transforms.append(v2.GaussianBlur(kernel_size=t.kernel_size, sigma=t.sigma))
            case "gaussian_noise":
                transforms.append(AddGaussianNoise(mean=t.mean, sigma=t.sigma))
            case _:
                raise ValueError(f"Unknown transform: {t.name}")

    return v2.Compose(transforms)


def build_dataset(cfg, segments, transform=None, use_pseudo_labels=False, stride=None):
    return InkDataset(
        cfg.path.data_dir,
        segments=segments,
        tile_size=cfg.dataset.tile_size,
        stride=cfg.dataset.stride if stride is None else stride,
        n_layers=cfg.dataset.n_layers,
        center=cfg.dataset.center,
        depth=cfg.dataset.depth,
        norm=cfg.dataset.norm,
        min_mask_fraction=cfg.dataset.min_mask_fraction,
        use_pseudo_labels=use_pseudo_labels,
        pseudo_label_version=None,
        transform=transform,
    )


def build_dataloader(cfg, use_unlabeled_data=False, aug=None):
    if aug is not None:
        transform = build_augmentation(aug)
    else:
        transform = None

    train_dataset = build_dataset(cfg, cfg.train.labeled, transform=transform, use_pseudo_labels=False)

    if use_unlabeled_data:
        pseudo_dataset = build_dataset(cfg, segments=cfg.train.unlabeled, transform=transform, use_pseudo_labels=True)
        train_dataset = CombinedInkDataset([train_dataset, pseudo_dataset])

    val_dataset = build_dataset(cfg, segments=cfg.train.val, use_pseudo_labels=False)

    sampler = create_balanced_sampler(
        train_dataset,
        mode=cfg.sampler.mode,
        num_samples=cfg.sampler.num_samples,
        alpha=cfg.sampler.alpha,
        pos_threshold=cfg.sampler.pos_threshold,
    )

    train_loader = create_dataloader(
        train_dataset,
        sampler=sampler,
        batch_size=cfg.dataloader.batch_size,
        shuffle=cfg.dataloader.shuffle,
        num_workers=cfg.dataloader.num_workers,
        pin_memory=cfg.dataloader.pin_memory,
        prefetch_factor=cfg.dataloader.prefetch_factor,
        persistent_workers=cfg.dataloader.persistent_workers,
    )

    val_loader = create_dataloader(
        val_dataset,
        sampler=None,
        batch_size=cfg.dataloader.batch_size,
        shuffle=False,
        num_workers=cfg.dataloader.num_workers,
        pin_memory=cfg.dataloader.pin_memory,
        prefetch_factor=cfg.dataloader.prefetch_factor,
        persistent_workers=cfg.dataloader.persistent_workers,
    )

    return train_loader, val_loader


def build_unlabeled_dataloader(cfg, aug=None, stride=None):
    if aug is not None:
        transform = build_augmentation(aug)
    else:
        transform = None

    pseudo_dataset = build_dataset(cfg, segments=cfg.train.unlabeled, transform=transform, stride=stride)

    pseudo_loader = create_dataloader(
        pseudo_dataset,
        sampler=None,
        batch_size=cfg.dataloader.batch_size,
        shuffle=False,
        num_workers=cfg.dataloader.num_workers,
        pin_memory=cfg.dataloader.pin_memory,
        prefetch_factor=cfg.dataloader.prefetch_factor,
        persistent_workers=cfg.dataloader.persistent_workers,
    )

    return pseudo_loader
