import os
import sys
import argparse
from pathlib import Path

import torch

from dotenv import load_dotenv
load_dotenv()

sys.path.append(os.getenv("CODE_DIR"))

from models.vesuvius_models.vesuvius_unet import VesuviusUNet
from models.vesuvius_models.vesuvius_unet_reference import VesuviusUnet_GitHub


def _get_arguments():
    parser = argparse.ArgumentParser(
        description="Convert all downloaded Zarr segments into TIFF stacks while preserving the directory structure."
    )

    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=None,
        help="Path to the checkpoint file.",
    )

    return parser.parse_args()


def main(args):
    original_model = VesuviusUnet_GitHub().cuda()
    model = VesuviusUNet(in_channels=1, out_channels=1, z_projection="max").cuda()

    if args.checkpoint_path is not None and args.checkpoint_path.exists():
        ckpt = torch.load(args.checkpoint_path, weights_only=False)
        original_model.load_vesuvius_checkpoint(ckpt)
        print("Original model checkpoint loaded")
        model.load_vesuvius_checkpoint(ckpt)
        print("Reimplemented model checkpoint loaded")
    else:
        model.load_vesuvius_checkpoint(original_model.get_state_dict())

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
