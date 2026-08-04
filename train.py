import os
from pathlib import Path

import torch
import hydra
from omegaconf import DictConfig, OmegaConf
import lightning as pl
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint

from data import build_dataloader
from models import build_model
from lightning_module import LitModel, DiceBCELoss

from dotenv import load_dotenv
load_dotenv()
for key in ["DATA_DIR", "CODE_DIR", "OUTPUT_DIR"]:
    if key in os.environ:
        os.environ[key] = os.path.expandvars(os.environ[key])


torch.set_float32_matmul_precision("high")


@hydra.main(version_base=None, config_path="conf/", config_name="-provide_as_argument-")
def main(cfg):
    output_dir = Path(cfg.path.output_dir)

    checkpoint_dir = output_dir / "checkpoints"
    lightning_dir = output_dir / "lightning"
    val_image_dir = output_dir / "val_images"

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    lightning_dir.mkdir(parents=True, exist_ok=True)
    val_image_dir.mkdir(parents=True, exist_ok=True)

    loader_train, loader_val = build_dataloader(cfg)

    model = build_model(cfg)
    loss = DiceBCELoss(w=cfg.loss.w, eps=cfg.loss.eps)

    lit_model = LitModel(
        model,
        loss,
        lr=cfg.train.lr,
        warmup_epochs=cfg.train.warmup_epochs,
        max_epochs=cfg.train.max_epochs,
        weight_decay=cfg.train.weight_decay,
        val_img_dir=val_image_dir,
        batch_size=cfg.dataloader.batch_size
    )

    logger = WandbLogger(project=cfg.wandb.project, name=cfg.experiment.name)
    logger.experiment.config.update(OmegaConf.to_container(cfg, resolve=True))

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
                filename=cfg.model.name + "_{epoch}_{val_loss:.3f}",
                every_n_epochs=cfg.train.checkpoint_every_n_epochs
            )
        ]
    )


    trainer.fit(
        lit_model,
        loader_train,
        loader_val
    )

if __name__ == "__main__":
    main()
