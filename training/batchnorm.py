import torch
from torch.optim.swa_utils import update_bn
from tqdm import tqdm

from .precision import autocast_context


@torch.no_grad()
def update_batchnorm_statistics(model, dataloader, device, autocast_dtype=None):
    """Adapt PyTorch's update_bn to tile batches, precision and inference noise."""

    batchnorms = [module for module in model.modules() if isinstance(module, torch.nn.modules.batchnorm._BatchNorm)]
    tracked_batchnorms = [module for module in batchnorms if module.track_running_stats]

    if not tracked_batchnorms:
        return

    training_states = {module: module.training for module in model.modules()}
    momenta = {module: module.momentum for module in batchnorms}
    parameter = next(model.parameters(), None)
    model_dtype = parameter.dtype if parameter is not None else tracked_batchnorms[0].running_mean.dtype

    def calibration_inputs():
        # update_bn calls model.train() before consuming this generator.
        # Re-enable only BatchNorm so dropout and DropPath stay disabled.

        model.eval()

        for module in tracked_batchnorms:
            module.train()

        num_batches = 0
        for batch in tqdm(dataloader, desc="  Calibrate BatchNorm", dynamic_ncols=True, leave=False):
            num_batches += 1
            yield batch["tile"].to(dtype=model_dtype, non_blocking=True)

        if num_batches == 0:
            raise ValueError("Cannot calibrate BatchNorm with an empty training dataloader.")

    try:
        with autocast_context(device.type, autocast_dtype):
            update_bn(calibration_inputs(), model, device=device)
    finally:
        # PyTorch restores these only after a successful pass and restores
        # training mode globally rather than preserving per-module states.

        for module, momentum in momenta.items():
            module.momentum = momentum
            
        for module, training in training_states.items():
            module.training = training
