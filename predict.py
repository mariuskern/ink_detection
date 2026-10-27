import os
import argparse
import pathlib
pathlib.PosixPath = pathlib.WindowsPath
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
from tqdm import tqdm
import hydra
from hydra import compose, initialize
from omegaconf import OmegaConf, DictConfig

from utils.predict import predict
from utils.save import save_predictions_as_images
from utils.device import get_device
from models import build_model
from data.augmentations import build_augmentation
from data.builder import build_dataset, build_dataloader

from dotenv import load_dotenv
load_dotenv()
for key in ["DATA_DIR", "CODE_DIR", "OUTPUT_DIR", "TMP_DIR"]:
    if key in os.environ:
        os.environ[key] = os.path.expandvars(os.environ[key])


# def parse_args():
#     parser = argparse.ArgumentParser(
#         description="Run model prediction on a dataset."
#     )

#     parser.add_argument(
#         "checkpoint",
#         type=Path,
#         help="Path to the checkpoint directory.",
#     )

#     return parser.parse_args()


@hydra.main(version_base=None, config_path="conf/", config_name="predict")
def main(cfg: DictConfig):
    # args = parse_args()

    # with initialize(config_path="conf/", version_base=None):
    #     cfg = compose(config_name="predict")

    # if not args.checkpoint.exists():
    #     raise FileNotFoundError(f"Checkpoint directory does not exist: {args.checkpoint}")

    if not cfg.checkpoint:
        raise ValueError("Checkpoint has to be given (checkpoint=<path/to/checkpoint.ckpt)")

    checkpoint = Path(cfg.checkpoint)

    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint directory does not exist: {checkpoint}")

    output_dir = Path(cfg.path.output_dir)
    tmp_dir = Path(cfg.path.tmp_dir)

    device = get_device(cfg)

    # model = build_model(cfg)
    # model = model.to(device)
    # # if cfg.train.compile:
    # #     model = torch.compile(model, mode=cfg.train.compile_mode)
    # ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)

    # state_dict = ckpt["state_dict"]

    # model_keys = model.state_dict().keys()
    # checkpoint_keys = state_dict.keys()

    # first_model_key = next(iter(model_keys))
    # first_checkpoint_key = next(iter(checkpoint_keys))

    # for prefix in ("model.", "_orig_mod."): # Remove model or _orig_mod (used with torch.compile) prefix
    #     if first_checkpoint_key.startswith(prefix) and not first_model_key.startswith(prefix):
    #         state_dict = {key.removeprefix(prefix): value for key, value in state_dict.items()}
    #         break

    # model.load_state_dict(state_dict)

    # transform = build_augmentation(cfg.aug) if cfg.aug is not None else None
    transforms = []
    if cfg.aug is not None:
        for variant in cfg.aug.variants:
            transforms.append(build_augmentation(variant))
    predict_dataset = build_dataset(cfg, cfg.train.segments, split="train" if cfg.train.use_validation_mask else None)
    predict_loader = build_dataloader(cfg, predict_dataset, None)

    # results = predict(model, predict_loader, device=device, tta_transform=transform, tta_num=cfg.train.tta_num)
    i = 0
    for result in predict(cfg, checkpoint, predict_loader, device=device, transforms=transforms, tta_num=cfg.train.tta_num, cache_dir=tmp_dir):
        save_predictions_as_images(
            output_dir,
            [result],
            save_probability=True,
            save_confidence=True,
            save_uncertainty=True,
            size=cfg.train.size,
        )

        i += 1

    print(f"Predicted {i} segments.")


if __name__ == "__main__":
    main()