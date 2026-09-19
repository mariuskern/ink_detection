import numpy as np
from dataclasses import dataclass
import numpy as np


@dataclass
class Prediction:
    segment: str

    probability: np.ndarray
    confidence: np.ndarray | None = None
    uncertainty: np.ndarray | None = None

    pseudo_label: np.ndarray | None = None
    pseudo_mask: np.ndarray | None = None
    
    pseudo_label_not_refined: np.ndarray | None = None
    pseudo_mask_not_refined: np.ndarray | None = None
    
    debug: dict | None = None


class PredictionMerger:
    def __init__(self):
        self.buffers = {}


    def add(self, segment, x, y, prediction, shape, target=None, valid_mask=None):
        if segment not in self.buffers:
            H, W = shape[1], shape[2]

            self.buffers[segment] = {
                "prob_sum": np.zeros((H, W), dtype=np.float32),
                "prob_sq_sum": np.zeros((H, W), dtype=np.float32),
                "weight_sum": np.zeros((H, W), dtype=np.float32),
                "target_sum": np.zeros((H, W), dtype=np.float32),
                "valid_weight_sum": np.zeros((H, W), dtype=np.float32),
            }

        buffer = self.buffers[segment]

        h, w = prediction.shape
        H, W = buffer["prob_sum"].shape

        h_valid = min(h, H - y)
        w_valid = min(w, W - x)

        if h_valid <= 0 or w_valid <= 0:
            return

        prediction = prediction[:h_valid, :w_valid]
        weight = self._gaussian_weight(h, w)[:h_valid, :w_valid]

        buffer["prob_sum"][y:y+h_valid, x:x+w_valid] += prediction * weight
        buffer["prob_sq_sum"][y:y+h_valid, x:x+w_valid] += (prediction ** 2) * weight
        buffer["weight_sum"][y:y+h_valid, x:x+w_valid] += weight
        if target is not None and valid_mask is not None:
            target = target[:h_valid, :w_valid]
            valid_mask = valid_mask[:h_valid, :w_valid]
            buffer["target_sum"][y:y+h_valid, x:x+w_valid] += target * valid_mask * weight
            buffer["valid_weight_sum"][y:y+h_valid, x:x+w_valid] += valid_mask * weight


    def get_predictions(self):
        results = []

        for segment, buffer in self.buffers.items():
            prediction, uncertainty, confidence = self._get_prediction(buffer)

            results.append(
                Prediction(
                    segment=segment,
                    probability=prediction,
                    uncertainty=uncertainty,
                    confidence=confidence,
                )
            )

        return results


    def iter_predictions(self):
        for segment, buffer in self.buffers.items():
            prediction, uncertainty, confidence = self._get_prediction(buffer)

            yield Prediction(
                segment=segment,
                probability=prediction,
                uncertainty=uncertainty,
                confidence=confidence,
            )


    def clear(self):
        self.buffers.clear()


    def get_binary_metrics(self, threshold=0.5):
        """Return a pixel-weighted Dice/IoU after overlapping tiles are merged."""

        intersection = prediction_sum = target_sum = 0.0

        for buffer in self.buffers.values():
            valid = buffer["valid_weight_sum"] > 0

            if not valid.any():
                continue

            # probability = buffer["prob_sum"] / np.maximum(buffer["weight_sum"], 1e-6)
            # target = buffer["target_sum"] / np.maximum(buffer["valid_weight_sum"], 1e-6)
            # prediction = probability >= threshold
            # target = target >= 0.5

            probability = buffer["prob_sum"] / np.maximum(buffer["weight_sum"], 1e-6)
            prediction = probability >= threshold
            target = buffer["target_sum"] >= 0.5 * np.maximum(buffer["valid_weight_sum"], 1e-6)

            intersection += np.logical_and(prediction, target)[valid].sum()
            prediction_sum += prediction[valid].sum()
            target_sum += target[valid].sum()

        if prediction_sum + target_sum == 0:
            return None
        
        dice = (2 * intersection) / (prediction_sum + target_sum)
        iou = intersection / (prediction_sum + target_sum - intersection)

        return {"dice": float(dice), "iou": float(iou)}


    def _get_prediction(self, buffer):
        prediction = buffer["prob_sum"] / np.maximum(buffer["weight_sum"], 1e-6)

        second_moment = buffer["prob_sq_sum"] / np.maximum(buffer["weight_sum"], 1e-6)
        variance = np.maximum(second_moment - prediction**2, 0)
        uncertainty = np.sqrt(variance)

        confidence = np.maximum(prediction, 1 - prediction)

        return prediction, uncertainty, confidence


    @staticmethod
    def _gaussian_weight(height, width, sigma_scale=1/6):
        y = np.arange(height) - (height - 1) / 2
        x = np.arange(width) - (width - 1) / 2

        xx, yy = np.meshgrid(x, y)

        sigma_y = height * sigma_scale
        sigma_x = width * sigma_scale

        g = np.exp(
            -(xx**2 / (2 * sigma_x**2)
            + yy**2 / (2 * sigma_y**2))
        )

        return (g / g.max()).astype(np.float32)


    # @staticmethod
    # def _gaussian_weight(size, sigma_scale=1/6):
    #     ax = np.arange(size) - (size - 1) / 2

    #     xx, yy = np.meshgrid(ax, ax)

    #     sigma = size * sigma_scale

    #     g = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))

    #     return (g / g.max()).astype(np.float32)
