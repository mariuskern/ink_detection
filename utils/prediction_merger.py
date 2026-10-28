import numpy as np
from dataclasses import dataclass
import shutil

from pathlib import Path


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
    # Segment metrics and a nested "global" dictionary for the full prediction run.
    metrics: dict[str, float | dict[str, float]] | None = None


class PredictionMerger:
    def __init__(self, cache_dir=None):
        self.buffers = {}

        self.cache_dir = Path(cache_dir) if cache_dir else None

        # if self.cache_dir:
        #     self.cache_dir.mkdir(parents=True, exist_ok=True)


    def _create_buffer(self, segment, shape):
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        H, W = shape[1], shape[2]

        if self.cache_dir is None:
            return {
                "prob_sum": np.zeros((H, W), dtype=np.float32),
                "prob_sq_sum": np.zeros((H, W), dtype=np.float32),
                "weight_sum": np.zeros((H, W), dtype=np.float32),
                "target_sum": np.zeros((H, W), dtype=np.float32),
                "valid_weight_sum": np.zeros((H, W), dtype=np.float32),
            }

        return {
            name: np.lib.format.open_memmap(
                self.cache_dir / f"{segment}_{name}.npy",
                mode="w+",
                dtype=np.float32,
                shape=(H, W),
            )
            for name in [
                "prob_sum",
                "prob_sq_sum",
                "weight_sum",
                "target_sum",
                "valid_weight_sum",
            ]
        }


    def add(self, segment, x, y, prediction, shape, target=None, valid_mask=None):
        if segment not in self.buffers:
            # H, W = shape[1], shape[2]

            # self.buffers[segment] = {
            #     "prob_sum": np.zeros((H, W), dtype=np.float32),
            #     "prob_sq_sum": np.zeros((H, W), dtype=np.float32),
            #     "weight_sum": np.zeros((H, W), dtype=np.float32),
            #     "target_sum": np.zeros((H, W), dtype=np.float32),
            #     "valid_weight_sum": np.zeros((H, W), dtype=np.float32),
            # }
            
            self.buffers[segment] = self._create_buffer(segment, shape)

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

        # print("input prediction:", prediction.dtype)
        # print("weight:", weight.dtype)
        # print("prob_sum:", buffer["prob_sum"].dtype)


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
        # Memmaps explizit schließen
        for buffer in self.buffers.values():
            for array in buffer.values():
                if isinstance(array, np.memmap):
                    array.flush()
                    array._mmap.close()
                    
        self.buffers.clear()

        # for segment in self.buffers:
        #     for name in ["prob_sum", "prob_sq_sum", "weight_sum", "target_sum", "valid_weight_sum"]:
        #         path = self.cache_dir / f"{segment}_{name}.npy"
        #         if path.exists():
        #             path.unlink()

        if self.cache_dir and self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)


    def get_binary_metrics(self, threshold=0.5):
        """Return merged binary metrics, or None without valid labels; zero denominators score 1."""

        intersection_sum = prediction_sum = target_sum = 0.0

        per_segment = {}

        for segment, buffer in self.buffers.items():
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

            intersection = np.logical_and(prediction, target)[valid].sum()
            prediction = prediction[valid].sum()
            target = target[valid].sum()

            intersection_sum += intersection
            prediction_sum += prediction
            target_sum += target

            denominator = prediction + target
            union = denominator - intersection

            dice_segment = (2 * intersection) / denominator if denominator > 0 else 1.0
            iou_segment = intersection / union if union > 0 else 1.0

            precision_segment = intersection / prediction if prediction > 0 else 1.0
            recall_segment = intersection / target if target > 0 else 1.0
            f0_5_denominator = prediction + 0.25 * target
            f0_5_segment = (1.25 * intersection) / f0_5_denominator if f0_5_denominator > 0 else 1.0
            f1_5_denominator = prediction + 2.25 * target
            f1_5_segment = (3.25 * intersection) / f1_5_denominator if f1_5_denominator > 0 else 1.0
            f2_denominator = prediction + 4 * target
            f2_segment = (5 * intersection) / f2_denominator if f2_denominator > 0 else 1.0

            per_segment[segment] = {
                "dice": float(dice_segment),
                "iou": float(iou_segment),
                "precision": float(precision_segment),
                "recall": float(recall_segment),
                "f0_5": float(f0_5_segment),
                "f1_5": float(f1_5_segment),
                "f2": float(f2_segment),
            }

        if not per_segment:
            return None
        
        denominator = prediction_sum + target_sum
        union = denominator - intersection_sum
        dice = (2 * intersection_sum) / denominator if denominator > 0 else 1.0
        iou = intersection_sum / union if union > 0 else 1.0
        
        precision = intersection_sum / prediction_sum if prediction_sum > 0 else 1.0
        recall = intersection_sum / target_sum if target_sum > 0 else 1.0
        f0_5_denominator = prediction_sum + 0.25 * target_sum
        f0_5 = (1.25 * intersection_sum) / f0_5_denominator if f0_5_denominator > 0 else 1.0
        f1_5_denominator = prediction_sum + 2.25 * target_sum
        f1_5 = (3.25 * intersection_sum) / f1_5_denominator if f1_5_denominator > 0 else 1.0
        f2_denominator = prediction_sum + 4 * target_sum
        f2 = (5 * intersection_sum) / f2_denominator if f2_denominator > 0 else 1.0

        return {
            "dice": float(dice),
            "iou": float(iou),
            "precision": float(precision),
            "recall": float(recall),
            "f0_5": float(f0_5),
            "f1_5": float(f1_5),
            "f2": float(f2),
            "per_segment": per_segment,
        }


    def _get_prediction(self, buffer):
        prediction = buffer["prob_sum"] / np.maximum(buffer["weight_sum"], 1e-6)

        # print("prob_sum:", buffer["prob_sum"].dtype)
        # print("weight_sum:", buffer["weight_sum"].dtype)
        # print("FINAL prediction:", prediction.dtype)

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
