import os
import sys
import argparse
from pathlib import Path

import torch
from transformers import AutoModel

from dotenv import load_dotenv
load_dotenv()

sys.path.append(os.getenv("CODE_DIR"))

from models.vesuvius_models import VesuviusResNet3D


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
    model_auto = AutoModel.from_pretrained("scrollprize/PHerc.1667-iteration-0", trust_remote_code=True).cuda()
    model = VesuviusResNet3D().cuda()

    if args.checkpoint_path is not None and args.checkpoint_path.exists():
        ckpt = torch.load(args.checkpoint_path, weights_only=False)
        model.load_state_dict(ckpt["state_dict"])
        print("Model checkpoint loaded")

    model_auto.eval()
    model.eval()

    for _ in range(10):
        x = torch.randn(4, 1, 64, 256, 256, device="cuda")
        
        print(x.shape)

        with torch.no_grad():
            y_auto = model_auto(x)["logits"]
            y_reimplemented = model(x)

        # print(y_auto.shape)
        # print(y_reimplemented.shape)

        print("max absolute difference:", (y_auto - y_reimplemented).abs().max().item())
        print("mean absolute difference:", (y_auto - y_reimplemented).abs().mean().item())
        print("allclose:", torch.allclose(y_auto, y_reimplemented, rtol=1e-5, atol=1e-6))


if __name__ == "__main__":
    main(_get_arguments())
