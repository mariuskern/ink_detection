import torch


def get_device(cfg):
    accelerator = str(cfg.train.accelerator).lower()

    if accelerator in {"gpu", "cuda"}:
        if not torch.cuda.is_available():
            raise RuntimeError("GPU/CUDA was requested, but CUDA is not available.")
        return torch.device("cuda")
    
    if accelerator == "cpu":
        return torch.device("cpu")

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")