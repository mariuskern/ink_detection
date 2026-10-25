import torch
from pathlib import Path

# from copy import deepcopy
# from omegaconf import open_dict
from omegaconf import OmegaConf

from models import build_model


def load_model_from_checkpoint(cfg, checkpoint_path: Path, model_dtype, device) -> torch.nn.Module:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # if "model_config" in checkpoint:
    #     cfg = deepcopy(cfg)
    #     with open_dict(cfg):
    #         cfg.model = checkpoint["model_config"]

    if "model_config" in checkpoint and OmegaConf.to_container(cfg.model, resolve=True) != checkpoint["model_config"]:
        raise ValueError("Model configuration mismatch: the current model config differs from the configuration stored in the checkpoint.")

    # Build a fresh, uncompiled model
    loaded_model = build_model(cfg)

    if model_dtype is not None:
        loaded_model = loaded_model.to(device=device, dtype=model_dtype)
    else:
        loaded_model = loaded_model.to(device)

    state_dict = checkpoint["state_dict"]

    # Remove torch.compile prefix if necessary
    if any(key.startswith("_orig_mod.") for key in state_dict):
        state_dict = {key.removeprefix("_orig_mod."): value for key, value in state_dict.items()}

    loaded_model.load_state_dict(state_dict)
    loaded_model.eval()

    del checkpoint
    del state_dict

    return loaded_model
