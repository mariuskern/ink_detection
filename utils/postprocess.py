import numpy as np


def postprocess_predictions(
    predictions,
    threshold,
    confidence_threshold,
    uncertainty_threshold,
    use_confidence_mask=True,
    use_uncertainty_mask=True,
):
    results = []

    for prediction in predictions:
        label = (prediction.probability >= threshold).astype(np.uint8)

        mask = np.ones_like(label, dtype=bool)

        if use_confidence_mask:
            mask &= prediction.confidence > confidence_threshold

        if use_uncertainty_mask:
            mask &= prediction.uncertainty < uncertainty_threshold

        prediction.pseudo_label = label
        prediction.pseudo_mask = mask

        results.append(prediction)

    return results
