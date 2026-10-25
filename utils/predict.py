from pathlib import Path

from tqdm import tqdm
import torch

from utils.prediction_merger import PredictionMerger
from utils.interpolate import interpolate_reverse
from training.precision import parse_precision, autocast_context
from utils.load_model import load_model_from_checkpoint


@torch.no_grad()
def predict(cfg, checkpoint_paths, dataloader, device, transforms=None, tta_num=1, cache_dir=None):
    if isinstance(checkpoint_paths, (str, Path)):
        checkpoint_paths = [checkpoint_paths]
    else:
        checkpoint_paths = list(checkpoint_paths)

    if len(checkpoint_paths) == 0:
        raise ValueError("checkpoint_paths must not be empty.")

    if device.type == "cuda":
        print(f"  CUDA memory: allocated={torch.cuda.memory_allocated(device) / 1024**3:.2f} GiB | reserved={torch.cuda.memory_reserved(device) / 1024**3:.2f} GiB")

    if tta_num < 1:
        raise ValueError("tta_num must be >= 1")

    # ---------------------------------------------------------
    # Normalize transforms
    # ---------------------------------------------------------
    if transforms is None:
        tta_transforms = [None]
    elif isinstance(transforms, (list, tuple)):
        tta_transforms = list(transforms)

        if len(tta_transforms) == 0:
            # raise ValueError("transforms must not be empty.")
            tta_transforms = [None]
    else:
        tta_transforms = [transforms]

    prediction_merger = PredictionMerger(cache_dir=cache_dir)

    autocast_dtype, _, model_dtype = parse_precision(cfg.train.precision)

    for path in checkpoint_paths:
        model = load_model_from_checkpoint(cfg, path, model_dtype=model_dtype, device=device)

        # model.to(device)
        model.eval()

        # ---------------------------------------------------------
        # TTA
        # ---------------------------------------------------------

        for tta_idx, tta_transform in enumerate(tta_transforms):
            repeats = 1 if tta_transform is None else tta_num

            for repeat_idx in range(repeats):
                # if tta_transform is None:
                # if tta_transform is None and len(tta_transforms) == 1:
                #     desc = "Predicting"
                # else:
                #     desc = f"TTA {tta_idx + 1}/{len(tta_transforms)} (repeat {repeat_idx + 1}/{repeats})"
                model_name = Path(path).stem
                if tta_transform is None and len(tta_transforms) == 1:
                    desc = f"Model {model_name}"
                else:
                    desc = f"Model {model_name} | TTA {tta_idx + 1}/{len(tta_transforms)} (repeat {repeat_idx + 1}/{repeats})"

                for batch in tqdm(dataloader, desc=f"  {desc}", dynamic_ncols=True, leave=False):
                    tile = batch["tile"]
                    segments = batch["segment"]
                    x = batch["x"]
                    y = batch["y"]
                    shape = batch["shape"]

                    tile = tile.to(device, non_blocking=True)

                    # -------------------------------------------------
                    # Apply transform
                    # -------------------------------------------------

                    if tta_transform is not None:
                        tile, params = tta_transform(tile, return_params=True)

                    # -------------------------------------------------
                    # Model
                    # -------------------------------------------------

                    # output = model(tile)
                    with autocast_context(device.type, autocast_dtype):
                        output = model(tile)

                    # ----------------------------------------------------------
                    # Resize model output back to transformed tile size
                    # ----------------------------------------------------------

                    needs_resize = tile.shape[-2:] != output.shape[-2:]
                    if needs_resize:
                        output = interpolate_reverse(output, tile.shape[-2:])

                    # ----------------------------------------------------------
                    # TTA inverse
                    # ----------------------------------------------------------

                    if tta_transform is not None:
                        output = tta_transform.inverse(output, params)

                    # -------------------------------------------------
                    # Probability
                    # -------------------------------------------------

                    # probs = torch.sigmoid(output).squeeze(1).cpu().numpy()
                    probs = torch.sigmoid(output).squeeze(1).float().cpu().numpy()

                    # -------------------------------------------------
                    # Prediction merger
                    # -------------------------------------------------
                    for i, segment in enumerate(segments):
                        prediction_merger.add(
                            segment=segment,
                            x=int(x[i]),
                            y=int(y[i]),
                            prediction=probs[i],
                            shape=shape[i].tolist(),
                        )

        del model

        if device.type == "cuda":
            torch.cuda.empty_cache()

    # ---------------------------------------------------------
    # Return merged predictions
    # ---------------------------------------------------------

    for prediction in prediction_merger.iter_predictions():
        yield prediction

    prediction_merger.clear()





# @torch.no_grad()
# def predict(model, dataloader, device):
#     # device = "cuda" if torch.cuda.is_available() else "cpu"

#     model.to(device)
#     model.eval()

#     prediction_merger = PredictionMerger()

#     for batch in tqdm(dataloader, desc="Predicting"):
#         tile = batch["tile"]
#         segments = batch["segment"]
#         x = batch["x"]
#         y = batch["y"]
#         shape = batch["shape"]

#         tile = tile.to(device, non_blocking=True)
#         output = model(tile)
#         probs = torch.sigmoid(output).squeeze(1).cpu().numpy()

#         for i, segment in enumerate(segments):
#             prediction_merger.add(
#                 segment=segment,
#                 x=int(x[i]),
#                 y=int(y[i]),
#                 prediction=probs[i],
#                 shape=shape[i].tolist(),
#             )

#     # return prediction_merger.get_predictions()
#     yield from prediction_merger.iter_predictions()
