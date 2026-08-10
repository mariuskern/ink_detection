from tqdm import tqdm
import torch

from .prediction_merger import PredictionMerger


@torch.no_grad()
def predict(model, dataloader, device=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model.to(device)
    model.eval()

    
    prediction_merger = PredictionMerger()

    for batch in tqdm(dataloader, desc="Predicting"):
        tile = batch["tile"]
        segments = batch["segment"]
        x = batch["x"]
        y = batch["y"]
        shape = batch["shape"]
    
        tile = tile.to(device, non_blocking=True)
        output = model(tile)
        probs = torch.sigmoid(output).squeeze(1).cpu().numpy()
    
        for i, segment in enumerate(segments):
            prediction_merger.add(
                segment=segment,
                x=int(x[i]),
                y=int(y[i]),
                prediction=probs[i],
                shape=shape[i].tolist(),
            )

    return prediction_merger.get_predictions()
    


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
