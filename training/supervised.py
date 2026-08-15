from pathlib import Path
from dataclasses import dataclass

import lightning as pl
import torch
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor

from data import build_dataloader
from data.builder import build_train_dataset, build_dataset, build_sampler, build_dataloader
from models import build_model, build_loss, LitModel


def train_supervised(cfg, logger):
    output_dir = Path(cfg.path.output_dir)

    fit(cfg, logger, output_dir)


@dataclass(frozen=True)
class TrainingResult:
    model: LitModel
    best_checkpoint: Path | None
    best_score: float | None


def fit(cfg, logger, output_dir, *, pseudo_label_version=None, augmentation=None):
    checkpoint_dir = output_dir / "checkpoints"
    val_image_dir = output_dir / "val_images"

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    val_image_dir.mkdir(parents=True, exist_ok=True)
    
    model = build_model(cfg)
    if cfg.train.compile:
        model = torch.compile(model, mode=cfg.train.compile_mode)
    loss = build_loss(cfg)

    lit_model = LitModel(
        model,
        loss,
        lr=cfg.train.lr,
        warmup_epochs=cfg.train.warmup_epochs,
        max_epochs=cfg.train.max_epochs,
        weight_decay=cfg.train.weight_decay,
        val_img_dir=val_image_dir,
        val_image_every_n_epochs=cfg.train.val_image_every_n_epochs,
        batch_size=cfg.dataloader.batch_size,
        iteration=str(pseudo_label_version) if pseudo_label_version is not None else None,
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="train/loss",
        mode="min",
        save_top_k=cfg.train.save_top_k,
        save_last=True,
        dirpath=checkpoint_dir,
        filename=cfg.model.name + "_{epoch:03d}_{val_loss:.3f}",
        every_n_epochs=cfg.train.checkpoint_every_n_epochs
    )

    trainer = pl.Trainer(
        max_epochs=cfg.train.max_epochs,
        accelerator=cfg.train.accelerator,
        devices=cfg.train.devices,
        logger=logger,
        # default_root_dir=lightning_dir,
        gradient_clip_val=cfg.train.gradient_clip_val,
        gradient_clip_algorithm=cfg.train.gradient_clip_algorithm,
        log_every_n_steps=10,
        precision=cfg.train.precision,
        benchmark=cfg.train.benchmark,
        deterministic=cfg.train.deterministic,
        callbacks=[
            checkpoint_callback,
            LearningRateMonitor(logging_interval="step")
        ]
    )

    train_dataset = build_train_dataset(cfg, pseudo_label_version=pseudo_label_version, augmentation=augmentation)
    val_dataset = build_dataset(cfg, cfg.train.val, min_mask_fraction=0.0)
    train_sampler = build_sampler(cfg, train_dataset)
    train_loader = build_dataloader(cfg, train_dataset, train_sampler)
    val_loader = build_dataloader(cfg, val_dataset, None)

    trainer.fit(lit_model, train_loader, val_loader)

    if checkpoint_callback.best_model_path:
        checkpoint = torch.load(checkpoint_callback.best_model_path, map_location="cpu", weights_only=False)
        lit_model.load_state_dict(checkpoint["state_dict"])
        best_checkpoint = Path(checkpoint_callback.best_model_path)
    else:
        best_checkpoint = None

    best_score = checkpoint_callback.best_model_score

    return TrainingResult(
        model=lit_model,
        best_checkpoint=best_checkpoint,
        best_score=best_score.item() if best_score is not None else None,
    )

