import torch


def calculate_dice(logits, target, valid_mask, threshold=0.5):
    prediction = (torch.sigmoid(logits) >= threshold).to(target.dtype) * valid_mask
    target = target * valid_mask
    intersection = (prediction * target).sum()
    dice = (2 * intersection + 1e-6) / (prediction.sum() + target.sum() + 1e-6)
    return dice


def calculate_iou(logits, target, valid_mask, threshold=0.5):
    prediction = (torch.sigmoid(logits) >= threshold).to(target.dtype) * valid_mask
    target = target * valid_mask
    intersection = (prediction * target).sum()
    union = prediction.sum() + target.sum() - intersection
    iou = (intersection + 1e-6) / (union + 1e-6)
    return iou