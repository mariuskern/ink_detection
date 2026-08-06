import torch
import torch.nn as nn


def build_loss(cfg):
    match cfg.loss.name:
        case "dice_bce":
            return DiceBCELoss(w=cfg.loss.w, eps=cfg.loss.eps)


class DiceBCELoss(nn.Module):
    def __init__(self, w=0.5, eps=1e-6):
        super().__init__()

        self.w = w
        self.eps = eps

        self.bce = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, logits, targets, mask=None):
        """
        Args:
            logits: Modelloutput, Shape (B,) oder (B,1)
            targets: Labels, Shape (B,) oder (B,1), Werte 0/1
        """

        # gleiche Shape erzwingen
        # logits = logits.view(-1)
        # targets = targets.float().view(-1)
        targets = targets.float()

        # BCE
        bce_loss = self.bce(logits, targets)

        # Dice
        probs = torch.sigmoid(logits)

        if mask is not None:
            mask = mask.float()
            bce_loss = bce_loss * mask
            probs = probs * mask
            targets = targets * mask

        if mask is not None:
            bce_loss = bce_loss.sum() / mask.sum().clamp(min=1)
        else:
            bce_loss = bce_loss.mean()

        intersection = (probs * targets).sum()

        dice_score = (2 * intersection + self.eps) / (probs.sum() + targets.sum() + self.eps)

        dice_loss = 1 - dice_score

        return self.w * bce_loss + (1 - self.w) * dice_loss
