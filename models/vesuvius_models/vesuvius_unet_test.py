import os
import sys
import argparse
from pathlib import Path

import torch

from dotenv import load_dotenv
load_dotenv()

sys.path.append(os.getenv("CODE_DIR"))

from models.vesuvius_models import VesuviusUNet
from models.vesuvius_models.challenge_code_5479453.config import TrainingConfig, resolve_training_mapping
from models.vesuvius_models.challenge_code_5479453.model import make_model


config = {
    "out_dir": "runs/ink_tutorial",
    "seed": 42,
    "mode": "flat",
    "model_type": "vesuvius_unet", # vesuvius_unet vesuvius_unet_2p5d
    "in_channels": 1,
    "model_config": {"autoconfigure": True, "z_projection_mode": "max"},
    "targets": {
        "ink": {"out_channels": 1, "activation": "none", "z_projection_mode": "max"}
    },
    "patch_size": [64, 256, 256],
    "patch_overlap": 0.5,
    "patch_min_labeled_coverage": 0.05,
    "batch_size": 2,
    "num_iterations": 20000,
    "learning_rate": 0.01,
    "mixed_precision": "fp16",
    "dataloader_workers": 4,
    "val_every": 500,
    "save_every": 1000,
    "verify_finite_gradients_steps": 200,
    "max_amp_overflow_events": 4,
    "datasets": [
        {
            "segments_path": "/path/to/ink-dataset/phercparis4",
            "segments": ["w00_20231016151002"],
            "volume_scale": "0",
        }
    ],
}

config = TrainingConfig.from_mapping(resolve_training_mapping(config))


def _get_arguments():
    parser = argparse.ArgumentParser(
        description="Convert all downloaded Zarr segments into TIFF stacks while preserving the directory structure."
    )

    parser.add_argument(
        "--checkpoint_path",
        type=Path,
        default=None,
        help="Path to the checkpoint file.",
    )

    return parser.parse_args()


def main(args):
    original_model = make_model(config.ink).cuda()
    model = VesuviusUNet(in_channels=1, out_channels=1, z_projection="max").cuda()

    if args.checkpoint_path is not None and args.checkpoint_path.exists():
        ckpt = torch.load(args.checkpoint_path, weights_only=False)
        original_model.load_state_dict(ckpt["model"])
        print("Original model checkpoint loaded")
        model.load_vesuvius_checkpoint(ckpt)
        print("Reimplemented model checkpoint loaded")
    else:
        model.load_vesuvius_checkpoint(original_model.state_dict())

    model.eval()
    original_model.eval()

    for _ in range(10):
        # x = torch.randn(4, 1, 64, 256, 256, device="cuda")
        x = torch.randn(4, 1, 32, 128, 128, device="cuda")

        # print(x.shape)

        with torch.no_grad():
            y_original = original_model(x)["ink"]
            y_reimplemented = model(x)

        # print(y_original.shape)
        # print(y_reimplemented.shape)

        print("max absolute difference:", (y_original - y_reimplemented).abs().max().item())
        print("mean absolute difference:", (y_original - y_reimplemented).abs().mean().item())
        print("allclose:", torch.allclose(y_original, y_reimplemented, rtol=1e-5, atol=1e-6))


if __name__ == "__main__":
    main(_get_arguments())
