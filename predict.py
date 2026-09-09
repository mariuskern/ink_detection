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
from hydra import compose, initialize
from omegaconf import DictConfig

from utils import predict, save_predictions_as_images
from utils.device import get_device
from models import build_model
from data.augmentations import build_augmentation
from data import build_dataset, build_dataloader

from dotenv import load_dotenv
load_dotenv()
for key in ["DATA_DIR", "CODE_DIR", "OUTPUT_DIR"]:
    if key in os.environ:
        os.environ[key] = os.path.expandvars(os.environ[key])


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run model prediction on a dataset."
    )

    parser.add_argument(
        "checkpoint",
        type=Path,
        help="Path to the checkpoint directory.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    with initialize(config_path="conf/", version_base=None):
        cfg = compose(config_name="predict")

    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint directory does not exist: {args.checkpoint}")

    output_dir = Path(cfg.path.output_dir)

    device = get_device(cfg)

    model = build_model(cfg)
    model = model.to(device)
    if cfg.train.compile:
        model = torch.compile(model, mode=cfg.train.compile_mode)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)

    state_dict = ckpt["state_dict"]

    model_keys = model.state_dict().keys()
    checkpoint_keys = state_dict.keys()

    first_model_key = next(iter(model_keys))
    first_checkpoint_key = next(iter(checkpoint_keys))

    if first_checkpoint_key.startswith("model.") and not first_model_key.startswith("model."):
        state_dict = {key.removeprefix("model."): value for key, value in state_dict.items()}
    model.load_state_dict(state_dict)

    transform = build_augmentation(cfg.aug) if cfg.aug is not None else None
    predict_dataset = build_dataset(cfg, cfg.train.segments, min_mask_fraction=0.0)
    predict_loader = build_dataloader(cfg, predict_dataset, None)

    results = predict(model, predict_loader, device=device, tta_transform=transform, tta_num=cfg.train.tta_num)

    save_predictions_as_images(
        output_dir,
        results,
        save_probability=True,
        save_confidence=True,
        save_uncertainty=True,
    )

    print(f"Predicted {len(results)} segments.")


if __name__ == "__main__":
    main()