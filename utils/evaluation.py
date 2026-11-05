import torch


COUNT_METRICS = ("tp", "fp", "tn", "fn")


@torch.no_grad()
def calculate_metrics(logits, target, valid_mask, threshold=0.5):
    """Keep existing (possibly soft) scores; counts use target >= 0.5 and mask > 0."""

    # prediction = torch.sigmoid(logits) >= threshold
    prediction = torch.sigmoid(logits.float()) >= threshold
    positive = target >= 0.5
    valid = valid_mask > 0

    tp = (prediction & positive & valid).sum()
    predicted_positive = (prediction & valid).sum()
    actual_positive = (positive & valid).sum()

    metrics = {
        "dice": calculate_dice(logits, target, valid_mask, threshold),
        "iou": calculate_iou(logits, target, valid_mask, threshold),
        **calculate_precision_recall_f(logits, target, valid_mask, threshold),
        "tp": tp,
        "fp": predicted_positive - tp,
        "tn": valid.sum() - predicted_positive - actual_positive + tp,
        "fn": actual_positive - tp,
    }

    # Preserve zero placeholders when the batch contains no samples of this kind.
    return metrics if logits.numel() else dict.fromkeys(metrics, 0.0)


def calculate_dice(logits, target, valid_mask, threshold=0.5):
    # prediction = (torch.sigmoid(logits.float()) >= threshold).to(target.dtype) * valid_mask
    prediction = (torch.sigmoid(logits.float()) >= threshold).to(target.dtype) * valid_mask
    target = target * valid_mask

    intersection = (prediction * target).sum()
    dice = (2 * intersection + 1e-6) / (prediction.sum() + target.sum() + 1e-6)
    
    return dice


def calculate_iou(logits, target, valid_mask, threshold=0.5):
    # prediction = (torch.sigmoid(logits) >= threshold).to(target.dtype) * valid_mask
    prediction = (torch.sigmoid(logits.float()) >= threshold).to(target.dtype) * valid_mask
    target = target * valid_mask

    intersection = (prediction * target).sum()
    union = prediction.sum() + target.sum() - intersection
    iou = (intersection + 1e-6) / (union + 1e-6)

    return iou


def calculate_precision_recall_f(logits, target, valid_mask, threshold=0.5, eps=1e-6):
    """Return masked Precision, Recall, F0.5, F1.5 and F2."""

    # prediction = (torch.sigmoid(logits) >= threshold).to(target.dtype)
    prediction = (torch.sigmoid(logits.float()) >= threshold).to(target.dtype)

    prediction = prediction * valid_mask
    target = target * valid_mask

    tp = (prediction * target).sum()
    predicted_positive = prediction.sum()
    actual_positive = target.sum()

    precision = (tp + eps) / (predicted_positive + eps)
    recall = (tp + eps) / (actual_positive + eps)
    f0_5 = (1.25 * tp + eps) / (predicted_positive + 0.25 * actual_positive + eps)
    f1_5 = (3.25 * tp + eps) / (predicted_positive + 2.25 * actual_positive + eps)
    f2 = (5 * tp + eps) / (predicted_positive + 4 * actual_positive + eps)

    return {
        "precision": precision,
        "recall": recall,
        "f0_5": f0_5,
        "f1_5": f1_5,
        "f2": f2,
    }
