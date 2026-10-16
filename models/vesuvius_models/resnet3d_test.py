import os
import sys
import argparse
from pathlib import Path

import torch

from dotenv import load_dotenv
load_dotenv()

sys.path.append(os.getenv("CODE_DIR"))

from models.vesuvius_models.resnet3d import VesuviusResNet3D
from models.vesuvius_models.resnet3d_reference import VesuviusResNet3D_GitHub, VesuviusResNet3D_HF


def _get_arguments():
    parser = argparse.ArgumentParser(
        description="Compare ResNet3D outputs with the GitHub and Hugging Face references."
    )

    parser.add_argument(
        "--checkpoint-path", type=Path, default=None,
        help="Checkpoint containing a state_dict. Without it, shared random weights are used.",
    )

    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu"
    )

    parser.add_argument(
        "--iterations",
        type=int,
        default=10
    )

    parser.add_argument(
        "--input_shape",
        type=int,
        nargs=5,
        default=(4, 1, 64, 256, 256),
        metavar=("B", "C", "D", "H", "W")
    )

    return parser.parse_args()


def main(args):
    torch.manual_seed(0)

    device = torch.device(args.device)

    model = VesuviusResNet3D(upscale=1).to(device)

    references = {
        "GitHub": VesuviusResNet3D_GitHub(upscale=1).to(device),
        "HF": VesuviusResNet3D_HF(
            in_channels=1,
            backbone_depth=50,
            backbone_channels=(256, 512, 1024, 2048),
            decoder_upscale=1,
        ).to(device),
    }

    if args.checkpoint_path is not None:
        checkpoint = torch.load(args.checkpoint_path, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["state_dict"])
        print(f"Checkpoint loaded: {args.checkpoint_path}")
    else:
        print("Using shared random weights (no checkpoint supplied).")

    state_dict = model.state_dict()
    references["GitHub"].load_state_dict(state_dict)
    # HF omits the unused classification head; all other keys must match exactly.
    hf_state_dict = {
        key: value for key, value in state_dict.items()
        if key not in {"backbone.fc.weight", "backbone.fc.bias"}
    }
    references["HF"].load_state_dict(hf_state_dict)

    model.eval()
    for reference in references.values():
        reference.eval()

    failures = []
    with torch.inference_mode():
        for iteration in range(args.iterations):
            x = torch.randn(*args.input_shape, device=device)

            actual = model(x)

            print(f"Iteration {iteration + 1}: input={tuple(x.shape)}, ResNet3D={tuple(actual.shape)}")

            for name, reference in references.items():
                expected = reference(x)

                print(f"  {name}: output={tuple(expected.shape)}")

                if actual.shape != expected.shape:
                    failures.append(f"Iteration {iteration + 1}: {name} output shape differs")
                    continue

                difference = (actual - expected).abs()

                matches = torch.allclose(actual, expected, rtol=1e-5, atol=1e-6)

                print(f"    max absolute difference: {difference.max().item()}")
                print(f"    mean absolute difference: {difference.mean().item()}")
                print(f"    allclose: {matches}")

                if not matches:
                    failures.append(f"Iteration {iteration + 1}: {name} outputs differ")

    print(f"\nNumber of errors: {len(failures)}")

    if failures:
        raise AssertionError("\n".join(failures))


if __name__ == "__main__":
    main(_get_arguments())
