# import torch
# from torchvision.transforms import v2

from .dataset import InkDataset, CombinedInkDataset
from .sampler import create_sampler
from .loader import create_dataloader
from .augmentations import build_augmentation


# class AddGaussianNoise(torch.nn.Module):
#     def __init__(self, mean=0., sigma=0.05):
#         super().__init__()

#         self.mean = mean
#         self.sigma = sigma

#     def forward(self, image, *masks):
#         noise = torch.randn_like(image)
#         image = image + noise * self.sigma + self.mean
#         image = torch.clamp(image, 0, 1)
#         return (image, *masks)


# def build_augmentation(aug):
#     transforms = []

#     def add_with_probability(transform, spec):
#         """Apply non-flip transforms only with the probability from the YAML."""
#         p = getattr(spec, "p", 1.0)
#         transforms.append(v2.RandomApply([transform], p=p) if p < 1.0 else transform)

#     for t in aug.transforms:
#         match t.name:
#             case "random_horizontal_flip":
#                 transforms.append(v2.RandomHorizontalFlip(p=t.p))
#             case "random_vertical_flip":
#                 transforms.append(v2.RandomVerticalFlip(p=t.p))
#             case "random_rotation":
#                 add_with_probability(v2.RandomRotation(degrees=t.degrees), t)
#             case "random_affine":
#                 add_with_probability(v2.RandomAffine(degrees=t.degrees, translate=t.translate, scale=t.scale, shear=t.shear), t)
#             case "random_resized_crop":
#                 add_with_probability(v2.RandomResizedCrop(size=t.size, scale=t.scale, ratio=t.ratio), t)
#             case "color_jitter":
#                 add_with_probability(v2.ColorJitter(brightness=t.brightness, contrast=t.contrast, saturation=t.saturation, hue=t.hue), t)
#             case "gaussian_blur":
#                 add_with_probability(v2.GaussianBlur(kernel_size=t.kernel_size, sigma=t.sigma), t)
#             case "gaussian_noise":
#                 add_with_probability(AddGaussianNoise(mean=t.mean, sigma=t.sigma), t)
#             case _:
#                 raise ValueError(f"Unknown transform: {t.name}")

#     if not transforms:
#         return v2.Identity()

#     return v2.Compose(transforms)


def build_dataset(
    cfg,
    segments,
    *,
    transform=None,
    use_as_unlabeled=False,
    pseudo_label_version=None,
    stride=None,
    min_mask_fraction=None,
    split=None
):
    use_pseudo_labels = pseudo_label_version is not None

    return InkDataset(
        cfg.path.data_dir,
        segments=segments,
        segment_crops=cfg.train.segment_crops,
        tile_size=cfg.dataset.tile_size,
        stride=cfg.dataset.stride if stride is None else stride,
        n_layers=cfg.dataset.n_layers,
        center=cfg.dataset.center,
        depth=cfg.dataset.depth,
        norm=cfg.dataset.norm,
        min_mask_fraction=cfg.dataset.min_mask_fraction if min_mask_fraction is None else min_mask_fraction,
        use_as_unlabeled=use_as_unlabeled,
        pseudo_label_version=pseudo_label_version,
        pseudo_target=cfg.train.pseudo_target if use_pseudo_labels else "hard",
        transform=transform,
        split=split
    )


def build_train_dataset(cfg, use_unlabeled_data=False, pseudo_label_version=None, augmentation=None, split=None):
    transform = build_augmentation(augmentation) if augmentation is not None else None

    dataset = build_dataset(cfg, cfg.train.labeled, transform=transform, split=split)

    if not use_unlabeled_data:
        return dataset

    pseudo_dataset = build_dataset(
        cfg,
        cfg.train.unlabeled,
        transform=transform,
        use_as_unlabeled=use_unlabeled_data,
        pseudo_label_version=pseudo_label_version,
        split=split
    )

    return CombinedInkDataset([dataset, pseudo_dataset])


# def build_val_dataset(cfg):
#     return build_dataset(cfg, cfg.train.val, min_mask_fraction=0.0)


# def build_predict_dataset(cfg, segments):
#     return build_dataset(cfg, segments, min_mask_fraction=0.0)


def build_sampler(cfg, dataset):
    return create_sampler(
        dataset,
        mode=cfg.sampler.mode,
        num_samples=cfg.sampler.num_samples,
        alpha=cfg.sampler.alpha,
        pos_threshold=cfg.sampler.pos_threshold,
        # real_fraction=cfg.sampler.real_fraction,
    )


def build_dataloader(cfg, dataset, sampler, shuffle=None):
    shuffle = cfg.dataloader.shuffle if shuffle is None else shuffle

    return create_dataloader(
        dataset,
        sampler=sampler,
        batch_size=cfg.dataloader.batch_size,
        shuffle=shuffle,
        num_workers=cfg.dataloader.num_workers,
        pin_memory=cfg.dataloader.pin_memory,
        prefetch_factor=cfg.dataloader.prefetch_factor,
        persistent_workers=cfg.dataloader.persistent_workers,
    )
