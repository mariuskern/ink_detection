from tqdm import tqdm
import torch

from .prediction_merger import PredictionMerger
from .interpolate import interpolate_reverse
from training.precision import parse_precision, autocast_context


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





@torch.no_grad()
def predict(model, dataloader, device, precision, transforms=None, tta_num=1, cache_dir=None):
    model.to(device)
    model.eval()

    if device.type == "cuda":
        print(f"[predict] CUDA memory allocated: {torch.cuda.memory_allocated(device) / 1024**3:.2f} GiB")
        print(f"[predict] CUDA memory reserved:  {torch.cuda.memory_reserved(device) / 1024**3:.2f} GiB")

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

    autocast_dtype, _, _ = parse_precision(precision)

    # ---------------------------------------------------------
    # TTA
    # ---------------------------------------------------------

    for tta_idx, tta_transform in enumerate(tta_transforms):
        repeats = 1 if tta_transform is None else tta_num

        for repeat_idx in range(repeats):
            # if tta_transform is None:
            if tta_transform is None and len(tta_transforms) == 1:
                desc = "Predicting"
            else:
                desc = f"TTA {tta_idx + 1}/{len(tta_transforms)} (repeat {repeat_idx + 1}/{repeats})"

            for batch in tqdm(dataloader, desc=desc):
                tile = batch["tile"]
                segments = batch["segment"]
                x = batch["x"]
                y = batch["y"]
                shape = batch["shape"]

                tile = tile.to(device, non_blocking=True)

                # -------------------------------------------------
                # Apply transform
                # -------------------------------------------------

                # transform_states = []

                # if tta_transform is not None:
                #     transformed_tiles = []

                #     for sample in tile:
                #         # sample: (C, D, H, W)
                #         # transformed_sample: (C, D, H, W)

                #         transformed_sample, state = tta_transform(sample, return_params=True)
                #         transformed_tiles.append(transformed_sample)
                #         transform_states.append(state)

                #     tile = torch.stack(transformed_tiles, dim=0)

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

                # if tta_transform is not None:
                #     restored_outputs = []

                #     for transformed_output, state in zip(output, transform_states):
                #         # transformed_output: (C, D, H, W)
                #         # restored_output: (C, D, H, W)
                #         restored_output = tta_transform.inverse(transformed_output, state)
                #         restored_outputs.append(restored_output)

                #     output = torch.stack(restored_outputs, dim=0)

                if tta_transform is not None:
                    output = tta_transform.inverse(output, params)

                # -------------------------------------------------
                # Probability
                # -------------------------------------------------

                probs = torch.sigmoid(output).squeeze(1).cpu().numpy()

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

    # ---------------------------------------------------------
    # Return merged predictions
    # ---------------------------------------------------------

    for prediction in prediction_merger.iter_predictions():
        yield prediction

    prediction_merger.clear()


# class Predictor:
#     def __init__(self, model, device=None):
#         self.model = model
#         self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

#         self.model.to(self.device)
#         self.model.eval()

#     @torch.no_grad()
#     def predict(self, dataloader):
#         prediction_merger = PredictionMerger()

#         for batch in tqdm(dataloader, desc="Predicting"):
#             self._predict_batch(batch, prediction_merger)

#         return prediction_merger.get_predictions()

#     def _predict_batch(self, batch, prediction_merger):
#         tile = batch["tile"]
#         segments = batch["segments"]
#         x = batch["x"]
#         y = batch["y"]
#         shape = batch["shape"]

#         tile = tile.to(self.device)
#         output = self.model(tile)
#         probs = torch.sigmoid(output).squeeze(1).cpu().numpy()

#         for i, segment in enumerate(segments):
#             prediction_merger.add(
#                 segment=segment,
#                 x=int(x[i]),
#                 y=int(y[i]),
#                 prediction=probs[i],
#                 segment_shape=shape[i].tolist(),
#             )
