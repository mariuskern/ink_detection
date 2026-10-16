# NOTE: This model serves as a reference implementation for comparing the AI-reimplemented
# model against the original Vesuvius Challenge model. It uses parts of the original
# implementation (https://github.com/ScrollPrize/villa/commit/5479453a76f0db39a8d657434d3d4c4c517f7245)
# to preserve the original model architecture and behavior as closely as possible.


import torch

from models.vesuvius_models.challenge_code_5479453.config import TrainingConfig, resolve_training_mapping
from models.vesuvius_models.challenge_code_5479453.model import make_model


class VesuviusUnet_GitHub(torch.nn.Module):
    def __init__(self):
        super().__init__()

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

        self.model = make_model(config.ink).cuda()

    def forward(self, x):
        return self.model(x)

    def load_vesuvius_checkpoint(self, ckpt):
        self.model.load_state_dict(ckpt["model"])

    def get_state_dict(self):
        return self.model.state_dict()