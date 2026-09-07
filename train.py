import os
from pathlib import Path
import warnings
# warnings.simplefilter("error", FutureWarning)
# Suppress a harmless deprecation warning caused by MONAI's DiNTS module.
warnings.filterwarnings(
    "ignore",
    message=r"`torch\.jit\.interface` is deprecated\.",
    category=FutureWarning,
)

import torch
import hydra
# import lightning as pl
# from lightning.pytorch.loggers import WandbLogger
from omegaconf import OmegaConf

from training import train_supervised, train_noisy_student
from training.seed import seed_everything
from training.logging import WandbLogger

from dotenv import load_dotenv
load_dotenv()
for key in ["DATA_DIR", "CODE_DIR", "OUTPUT_DIR"]:
    if key in os.environ:
        os.environ[key] = os.path.expandvars(os.environ[key])


torch.set_float32_matmul_precision("high")


@hydra.main(version_base=None, config_path="conf/", config_name="-provide_as_argument-")
def main(cfg):
    # pl.seed_everything(cfg.train.seed, workers=True)

    if cfg.train.seed is not None:
        seed_everything(cfg.train.seed)
        print(f"Seed: {cfg.train.seed}")
    else:
        print("Seed: no seed used")

    # logger = WandbLogger(project=cfg.wandb.project, name=cfg.experiment.name)
    # logger.experiment.config.update(OmegaConf.to_container(cfg, resolve=True))

    # run = wandb.init(
    #     project=cfg.wandb.project,
    #     name=cfg.experiment.name,
    #     config=OmegaConf.to_container(cfg, resolve=True),
    # )

    logger = WandbLogger(
        project=cfg.wandb.project,
        name=cfg.experiment.name,
        config=OmegaConf.to_container(cfg, resolve=True),
    )
    
    match cfg.train.name:
        case "supervised":
            train_supervised(cfg, logger)
        case "noisy_student":
            train_noisy_student(cfg, logger)
        # case "unimatch":
        #     train_unimatch(cfg, logger)
        case _:
            raise ValueError(f"{cfg.train.name} not supported")


if __name__ == "__main__":
    main()
