import numpy as np
import torch
# from scipy import ndimage
# from skimage.morphology import remove_small_objects
import cv2


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
    """
    Remove connected components smaller than min_size.

    Optionally updates the pseudo mask so that removed pixels are also
    excluded from the training mask.
    """

    def __init__(self, min_size, connectivity=4, update_mask=True):
        super().__init__()

        if min_size < 1:
            raise ValueError("min_size must be at least 1")

        if connectivity not in (4, 8):
            raise ValueError("connectivity must be 4 or 8")

        self.min_size = min_size
        self.connectivity = connectivity
        self.update_mask = update_mask

    def forward(self, predictions):
        for prediction in predictions:
            # label = prediction.pseudo_label.astype(bool)
            label = prediction.pseudo_label.astype(np.uint8)

            # refined_label = remove_small_objects(label, min_size=self.max_size, connectivity=self.connectivity)

            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(label, connectivity=self.connectivity)

            areas = stats[:, cv2.CC_STAT_AREA]

            keep = areas >= self.min_size
            keep[0] = False  # Hintergrund nicht behalten

            refined_label = keep[labels].astype(np.uint8)

            if self.update_mask:
                removed = label & ~refined_label

                prediction.pseudo_mask[removed] = False

            prediction.pseudo_label = refined_label.astype(np.uint8)

        return predictions


class BinaryClosing(torch.nn.Module):
    """
    Apply morphological binary closing to the pseudo label.

    Optionally restricts closing to regions where the prediction probability
    exceeds a given threshold.

    Newly created pixels can optionally be added to the pseudo mask.
    """

    def __init__(self, kernel_size=3, iterations=1, probability_threshold=None, update_mask=False):
        super().__init__()

        if kernel_size < 1 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer")

        if iterations < 1:
            raise ValueError("iterations must be at least 1")

        if probability_threshold is not None and (not 0.0 <= probability_threshold <= 1.0):
            raise ValueError("probability_threshold must be between 0 and 1")

        self.kernel_size = kernel_size
        self.iterations = iterations
        self.probability_threshold = probability_threshold
        self.update_mask = update_mask

        self.kernel = np.ones((self.kernel_size, self.kernel_size), dtype=np.uint8)

    def forward(self, predictions):
        # structure = np.ones((self.kernel_size, self.kernel_size), dtype=bool)

        for prediction in predictions:
            # label = prediction.pseudo_label.astype(bool)
            label = prediction.pseudo_label.astype(np.uint8)

            # if self.probability_threshold is not None:
            #     probability_mask = prediction.probability >= self.probability_threshold
            #     closing_input = label & probability_mask
            # else:
            #     closing_input = label

            # refined_label = ndimage.binary_closing(closing_input, structure=structure, iterations=self.iterations)

            # if self.probability_threshold is not None:
            #     refined_label = label & ~probability_mask | refined_label & probability_mask

            # closed = ndimage.binary_closing(label, structure=structure, iterations=self.iterations)

            closed = cv2.morphologyEx(
                label,
                cv2.MORPH_CLOSE,
                self.kernel,
                iterations=self.iterations,
            )

            if self.probability_threshold is not None:
                refined_label = closed & (prediction.probability >= self.probability_threshold)
            else:
                refined_label = closed

            if self.update_mask:
                added = refined_label & ~label
                prediction.pseudo_mask[added] = True

            prediction.pseudo_label = refined_label.astype(np.uint8)

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
                        min_size=step.min_size,
                        connectivity=step.connectivity,
                        update_mask=step.update_mask,
                    )
                )

            case "binary_closing":
                steps.append(
                    BinaryClosing(
                        kernel_size=step.kernel_size,
                        iterations=step.iterations,
                        probability_threshold=step.probability_threshold,
                        update_mask=step.update_mask,
                    )
                )
            case _:
                raise ValueError(
                    f"Unknown label refinement step: {step.name}"
                )

    return LabelRefinementPipeline(steps, debug=cfg.debug)