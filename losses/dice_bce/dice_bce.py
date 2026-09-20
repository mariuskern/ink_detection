import torch
import torch.nn as nn


class DiceBCELoss(nn.Module):
    def __init__(self, w=0.5, eps=1e-6, bce_label_smoothing=0.0, dice_label_smoothing=0.0):
        super().__init__()

        if not 0.0 <= bce_label_smoothing <= 1.0:
            raise ValueError("bce_label_smoothing must be between 0 and 1")

        if not 0.0 <= dice_label_smoothing <= 1.0:
            raise ValueError("dice_label_smoothing must be between 0 and 1")

        self.w = w
        self.eps = eps

        self.bce_label_smoothing = bce_label_smoothing
        self.dice_label_smoothing = dice_label_smoothing

        self.bce = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, logits, targets, mask=None, sample_weight=None):
        """
        Args:
            logits: Modelloutput, Shape (B,) oder (B,1)
            targets: Labels, Shape (B,) oder (B,1), Werte 0/1
        """

        targets = targets.float()

        if logits.shape != targets.shape:
            raise ValueError(f"logits and targets must have the same shape, got {tuple(logits.shape)} and {tuple(targets.shape)}")

        if mask is None:
            valid = torch.ones_like(targets)
        else:
            if mask.shape != targets.shape:
                raise ValueError(f"mask and targets must have the same shape, got {tuple(mask.shape)} and {tuple(targets.shape)}")
            valid = mask.to(dtype=targets.dtype, device=targets.device)

        if sample_weight is None:
            sample_weight = torch.ones(logits.shape[0], device=logits.device, dtype=targets.dtype)
        else:
            if sample_weight.shape != (logits.shape[0],):
                raise ValueError(f"sample_weight must have shape ({logits.shape[0]},), got {tuple(sample_weight.shape)}")
            sample_weight = sample_weight.to(dtype=targets.dtype, device=targets.device)

        sample_weight_map = sample_weight.view(-1, *([1] * (logits.ndim - 1)))

        weighted_valid = valid * sample_weight_map

        # BCE
        bce_targets = targets
        if self.bce_label_smoothing > 0.0:
            bce_targets = bce_targets * (1.0 - self.bce_label_smoothing) + 0.5 * self.bce_label_smoothing
        bce_loss = self.bce(logits, bce_targets)
        # bce_loss = (bce_loss * valid).sum() / valid.sum().clamp(min=1)
        bce_loss = (bce_loss * weighted_valid).sum() / weighted_valid.sum().clamp(min=1)

        # Dice
        probs = torch.sigmoid(logits)
        dice_targets = targets
        if self.dice_label_smoothing > 0.0:
            dice_targets = dice_targets * (1.0 - self.dice_label_smoothing) + 0.5 * self.dice_label_smoothing
        probs = probs * valid
        dice_targets = dice_targets * valid

        # intersection = (probs * targets).sum()
        # dice_score = (2 * intersection + self.eps) / (probs.sum() + targets.sum() + self.eps)

        dims = tuple(range(1, logits.ndim))

        intersection = (probs * dice_targets).sum(dim=dims)
        union = (probs.sum(dim=dims) + dice_targets.sum(dim=dims))
        dice = ( 2 * intersection + self.eps) / ( union + self.eps)

        # dice_score = dice.mean()
        dice_score = (dice * sample_weight).sum() / sample_weight.sum().clamp(min=1e-12)

        dice_loss = 1 - dice_score

        return self.w * bce_loss + (1 - self.w) * dice_loss