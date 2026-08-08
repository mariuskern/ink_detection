import numpy as np


def postprocess_predictions(predictions, threshold, confidence_threshold, uncertainty_threshold):
    results = []

    for prediction in predictions:
        label = (prediction.probability >= threshold).astype(np.uint8)

        mask = (prediction.confidence > confidence_threshold) & (prediction.uncertainty < uncertainty_threshold)

        prediction.pseudo_label = label
        prediction.pseudo_mask = mask

        results.append(prediction)

    return results
