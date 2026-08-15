from pathlib import Path
from dataclasses import dataclass

import lightning as pl
import torch
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint

from data import build_dataset, build_train_dataset, build_sampler, build_dataloader, build_augmentation
from models import UniMatchModule, build_model, build_loss


@dataclass(frozen=True)
class UniMatchTrainingResult:
    model: UniMatchModule
    best_checkpoint: Path | None
    best_score: float | None


def train_unimatch(cfg, logger):
    output_dir = Path(cfg.path.output_dir)

    return fit(cfg, logger, output_dir)


def fit(cfg, logger, output_dir):
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    val_image_dir = output_dir / "val_images"
    val_image_dir.mkdir(parents=True, exist_ok=True)

    model = build_model(cfg)
    if cfg.train.compile:
        model = torch.compile(model, mode=cfg.train.compile_mode)
    loss = build_loss(cfg)

    weak_transform = build_augmentation(cfg.train.weak_aug)
    strong_transform_1 = build_augmentation(cfg.train.strong_aug)
    strong_transform_2 = build_augmentation(cfg.train.strong_aug)

    lit_model = UniMatchModule(
        model,
        loss,
        lr=cfg.train.lr,
        warmup_epochs=cfg.train.warmup_epochs,
        max_epochs=cfg.train.max_epochs,
        weight_decay=cfg.train.weight_decay,
        val_img_dir=val_image_dir,
        val_image_every_n_epochs=cfg.train.val_image_every_n_epochs,
        batch_size=cfg.dataloader.batch_size,
        backbone_lr_multiplier=cfg.train.backbone_lr_multiplier,
        threshold=cfg.train.threshold,
        confidence_threshold=cfg.train.confidence_threshold,
        lambda_u=cfg.train.lambda_u,
        strong_loss_weight=cfg.train.strong_loss_weight,
        fp_loss_weight=cfg.train.fp_loss_weight,
        weak_transform=weak_transform,
        strong_transform_1=strong_transform_1,
        strong_transform_2=strong_transform_2,
        # use_cutmix=cfg.train.use_cutmix,
        # cutmix_alpha=cfg.train.cutmix_alpha,
        # cutmix_probability=cfg.train.cutmix_probability,
        feature_dropout_p=cfg.train.feature_dropout_p,
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="val/dice",
        mode="max",
        save_top_k=cfg.train.save_top_k,
        save_last=True,
        dirpath=checkpoint_dir,
        filename=cfg.model.name + "_{epoch:03d}_{val/dice:.3f}",
        every_n_epochs=cfg.train.checkpoint_every_n_epochs,
    )

    trainer = pl.Trainer(
        max_epochs=cfg.train.max_epochs,
        accelerator=cfg.train.accelerator,
        devices=cfg.train.devices,
        logger=logger,
        gradient_clip_val=cfg.train.gradient_clip_val,
        gradient_clip_algorithm=cfg.train.gradient_clip_algorithm,
        log_every_n_steps=10,
        precision=cfg.train.precision,
        benchmark=cfg.train.benchmark,
        deterministic=cfg.train.deterministic,
        callbacks=[
            checkpoint_callback,
            LearningRateMonitor(logging_interval="step"),
        ],
    )

    train_dataset = build_train_dataset(cfg, use_unlabeled_data=True)
    val_dataset = build_dataset(cfg, cfg.train.val, min_mask_fraction=0.0)
    train_sampler = build_sampler(cfg, train_dataset)
    train_loader = build_dataloader(cfg, train_dataset, train_sampler)
    val_loader = build_dataloader(cfg, val_dataset, None)

    trainer.fit(lit_model, train_loader, val_loader)

    if checkpoint_callback.best_model_path:
        checkpoint = torch.load(
            checkpoint_callback.best_model_path,
            map_location="cpu",
            weights_only=False,
        )
        lit_model.load_state_dict(checkpoint["state_dict"])
        best_checkpoint = Path(checkpoint_callback.best_model_path)
    else:
        best_checkpoint = None

    best_score = checkpoint_callback.best_model_score

    return UniMatchTrainingResult(
        model=lit_model,
        best_checkpoint=best_checkpoint,
        best_score=best_score.item() if best_score is not None else None,
    )
