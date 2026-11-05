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
from hydra.core.hydra_config import HydraConfig
from utils.config_snapshot import save_training_config, register_config_resolvers, resolve_resume_run, save_wandb_identity, prepare_teacher_source
from utils.teacher_checkpoints import resolve_teacher_checkpoint_groups

from training.supervised import train_supervised
from training.noisy_student import train_noisy_student
from training.seed import seed_everything
from training.utils import configure_determinism
from training.logging import WandbLogger

from dotenv import load_dotenv
load_dotenv()
for key in ["DATA_DIR", "CODE_DIR", "OUTPUT_DIR", "TMP_DIR"]:
    if key in os.environ:
        os.environ[key] = os.path.expandvars(os.environ[key])


torch.set_float32_matmul_precision("high")
register_config_resolvers()


@hydra.main(version_base=None, config_path="conf/", config_name="-provide_as_argument-")
def main(cfg):
    configure_determinism(
        deterministic=cfg.train.deterministic,
        benchmark=cfg.train.benchmark,
    )

    prepare_teacher_source(cfg)

    resume_run = resolve_resume_run(cfg)

    start_iteration = cfg.train.get("start_iteration", 0)
    is_noisy_student = cfg.train.name == "noisy_student"

    if resume_run and (not is_noisy_student or not 0 < start_iteration < cfg.train.iterations):
        raise ValueError("train.resume_run requires noisy_student with 0 < start_iteration < iterations")
        
    if is_noisy_student and start_iteration > 0:
        resolve_teacher_checkpoint_groups(cfg.train.get("teacher_checkpoint"), start_iteration, cfg.train.teacher, cfg.train.checkpoint, resume=bool(resume_run))

    save_training_config(cfg, HydraConfig.get())

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
        resume_run=resume_run,
        start_iteration=cfg.train.get("start_iteration", 0),
    )
    
    save_wandb_identity(cfg, HydraConfig.get(), logger.run)
    
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
