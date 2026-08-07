from pathlib import Path

import lightning as pl
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from omegaconf import OmegaConf

from data import build_dataloader
from models import build_model, build_loss
from lightning_module import LitModel


def train_one_iteration(cfg, logger, output_dir, aug, use_unlabeled_data=False, iteration=None):
    checkpoint_dir = output_dir / "checkpoints"
    lightning_dir = output_dir / "lightning"
    val_image_dir = output_dir / "val_images"

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    lightning_dir.mkdir(parents=True, exist_ok=True)
    val_image_dir.mkdir(parents=True, exist_ok=True)
    
    model = build_model(cfg)
    loss = build_loss(cfg)

    lit_model = LitModel(
        model,
        loss,
        lr=cfg.train.lr,
        warmup_epochs=cfg.train.warmup_epochs,
        max_epochs=cfg.train.max_epochs,
        weight_decay=cfg.train.weight_decay,
        val_img_dir=val_image_dir,
        batch_size=cfg.dataloader.batch_size,
        iteration=iteration,
    )

    trainer = pl.Trainer(
        max_epochs=cfg.train.max_epochs,
        accelerator=cfg.train.accelerator,
        devices=cfg.train.devices,
        logger=logger,
        default_root_dir=lightning_dir,
        gradient_clip_val=cfg.train.gradient_clip_val,
        gradient_clip_algorithm=cfg.train.gradient_clip_algorithm,
        log_every_n_steps=10,
        precision=cfg.train.precision,
        callbacks=[
            ModelCheckpoint(
                monitor="train/loss",
                mode="min",
                save_top_k=-1,
                dirpath=checkpoint_dir,
                filename=cfg.model.name + "_{epoch}_{train/loss:.3f}",
                every_n_epochs=cfg.train.checkpoint_every_n_epochs
            ),
            LearningRateMonitor(logging_interval="step")
        ]
    )

    loader_train, loader_val = build_dataloader(cfg, use_unlabeled_data=use_unlabeled_data, aug=aug)

    trainer.fit(lit_model, loader_train, loader_val)

    return lit_model
