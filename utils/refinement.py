import numpy as np
import torch
from scipy import ndimage
from skimage.morphology import remove_small_objects


class LabelRefinementPipeline:
    """Sequential pipeline for pseudo-label refinement."""

    def __init__(self, steps, debug=False):
        self.steps = steps
        self.debug = debug

    def __call__(self, predictions):
        is_single = not isinstance(predictions, (list, tuple))

        if is_single:
            predictions = [predictions]
        
        for i, step in enumerate(self.steps):
            predictions = step(predictions)

            if self.debug:
                step_name = f"{i:02d}_{step.__class__.__name__}"

                for prediction in predictions:
                    if not hasattr(prediction, "refinement_debug"):
                        prediction.refinement_debug = {}

                    prediction.refinement_debug[step_name] = {
                        "pseudo_label": prediction.pseudo_label.copy(),
                        "pseudo_mask": prediction.pseudo_mask.copy(),
                        "statistics": self._get_debug_statistics(prediction),
                    }

        if is_single:
            return predictions[0]

        return predictions

    def _get_debug_statistics(self, prediction):
        label = prediction.pseudo_label
        mask = prediction.pseudo_mask

        return {
            "label_pixels": int(label.sum()),
            "valid_pixels": int(mask.sum()),
            "total_pixels": int(mask.size),
            "valid_ratio": float(mask.mean()),
        }


class Threshold(torch.nn.Module):
    """Create pseudo labels and initialize the corresponding mask."""

    def __init__(self, threshold):
        super().__init__()
        self.threshold = threshold

    def forward(self, predictions):
        for prediction in predictions:
            prediction.pseudo_label = (prediction.probability >= self.threshold).astype(np.uint8)
            prediction.pseudo_mask = np.ones_like(prediction.pseudo_label, dtype=bool)
            prediction.pseudo_label_not_refined = prediction.pseudo_label
            prediction.pseudo_mask_not_refined = prediction.pseudo_mask

        return predictions


class ConfidenceMask(torch.nn.Module):
    """Mask predictions below a confidence threshold."""

    def __init__(self, threshold):
        super().__init__()
        self.threshold = threshold

    def forward(self, predictions):
        for prediction in predictions:
            mask = prediction.confidence > self.threshold

            if prediction.pseudo_mask is None:
                prediction.pseudo_mask = mask
            else:
                prediction.pseudo_mask &= mask

        return predictions


class UncertaintyMask(torch.nn.Module):
    """Mask predictions above an uncertainty threshold."""

    def __init__(self, threshold):
        super().__init__()
        self.threshold = threshold

    def forward(self, predictions):
        for prediction in predictions:
            mask = prediction.uncertainty < self.threshold

            if prediction.pseudo_mask is None:
                prediction.pseudo_mask = mask
            else:
                prediction.pseudo_mask &= mask

        return predictions


class RemoveSmallObjects(torch.nn.Module):
    """Remove connected components smaller than min_size."""

    def __init__(self, max_size, connectivity=1):
        super().__init__()

        if max_size < 1:
            raise ValueError("min_size must be at least 1")

        self.max_size = max_size
        self.connectivity = connectivity

    def forward(self, predictions):
        for prediction in predictions:
            label = prediction.pseudo_label.astype(bool)

            label = remove_small_objects(label, max_size=self.max_size, connectivity=self.connectivity)

            prediction.pseudo_label = label.astype(np.uint8)

        return predictions


class BinaryClosing(torch.nn.Module):
    """Apply morphological binary closing to the pseudo label."""

    def __init__(self, kernel_size=3, iterations=1):
        super().__init__()

        if kernel_size < 1 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer")

        if iterations < 1:
            raise ValueError("iterations must be at least 1")

        self.kernel_size = kernel_size
        self.iterations = iterations

    def forward(self, predictions):
        structure = np.ones((self.kernel_size, self.kernel_size), dtype=bool)

        for prediction in predictions:
            label = prediction.pseudo_label.astype(bool)

            label = ndimage.binary_closing(label, structure=structure, iterations=self.iterations)

            prediction.pseudo_label = label.astype(np.uint8)

        return predictions


def build_label_refinement(cfg):
    steps = []

    for step in cfg.steps:
        match step.name:
            case "threshold":
                steps.append(
                    Threshold(
                        threshold=step.threshold,
                    )
                )

            case "confidence_mask":
                steps.append(
                    ConfidenceMask(
                        threshold=step.threshold,
                    )
                )
            case "uncertainty_mask":
                steps.append(
                    UncertaintyMask(
                        threshold=step.threshold,
                    )
                )
            case "remove_small_objects":
                steps.append(
                    RemoveSmallObjects(
                        max_size=step.max_size,
                        connectivity=step.connectivity,
                    )
                )
            case "binary_closing":
                steps.append(
                    BinaryClosing(
                        kernel_size=step.kernel_size,
                        iterations=step.iterations,
                    )
                )
            case _:
                raise ValueError(
                    f"Unknown label refinement step: {step.name}"
                )

    return LabelRefinementPipeline(steps, debug=cfg.debug)