import os
from pathlib import Path

import torch
import hydra
from lightning.pytorch.loggers import WandbLogger
from omegaconf import OmegaConf

from training import train_supervised, train_noisy_student

from dotenv import load_dotenv
load_dotenv()
for key in ["DATA_DIR", "CODE_DIR", "OUTPUT_DIR"]:
    if key in os.environ:
        os.environ[key] = os.path.expandvars(os.environ[key])


torch.set_float32_matmul_precision("high")


# import sys

# print("None in sys.path:", None in sys.path)
# for i, p in enumerate(sys.path):
#     print(i, repr(p), type(p))


@hydra.main(version_base=None, config_path="conf/", config_name="-provide_as_argument-")
def main(cfg):
    logger = WandbLogger(project=cfg.wandb.project, name=cfg.experiment.name)
    logger.experiment.config.update(OmegaConf.to_container(cfg, resolve=True))
    
    match cfg.train.name:
        case "supervised":
            train_supervised(cfg, logger)
        case "noisy_student":
            train_noisy_student(cfg, logger)


if __name__ == "__main__":
    main()
