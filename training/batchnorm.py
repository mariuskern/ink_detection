import torch
from tqdm import tqdm

from .precision import autocast_context


@torch.no_grad()
def update_batchnorm_statistics(model, dataloader, device, autocast_dtype=None):
    """Recompute tracked BatchNorm statistics with dropout and DropPath disabled."""

    batchnorms = [module for module in model.modules() if isinstance(module, torch.nn.modules.batchnorm._BatchNorm) and module.track_running_stats]

    if not batchnorms:
        return

    training_states = {module: module.training for module in model.modules()}
    momenta = {module: module.momentum for module in batchnorms}
    parameter = next(model.parameters(), None)
    model_dtype = parameter.dtype if parameter is not None else batchnorms[0].running_mean.dtype
    num_batches = 0

    try:
        model.eval()

        for module in batchnorms:
            module.reset_running_stats()
            module.momentum = None
            module.train()

        for batch in tqdm(dataloader, desc="  Calibrate BatchNorm", dynamic_ncols=True, leave=False):
            tile = batch["tile"].to(device=device, dtype=model_dtype, non_blocking=True)

            with autocast_context(device.type, autocast_dtype):
                model(tile)
                
            num_batches += 1

        if num_batches == 0:
            raise ValueError("Cannot calibrate BatchNorm with an empty training dataloader.")
    finally:
        for module, momentum in momenta.items():
            module.momentum = momentum
        for module, training in training_states.items():
            module.training = training
