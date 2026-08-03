import torch
import torch.nn as nn


class DiceBCELoss(nn.Module):
    def __init__(self, w=0.5, eps=1e-6):
        super().__init__()

        self.w = w
        self.eps = eps

        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits, targets):
        """
        Args:
            logits: Modelloutput, Shape (B,) oder (B,1)
            targets: Labels, Shape (B,) oder (B,1), Werte 0/1
        """

        # gleiche Shape erzwingen
        logits = logits.view(-1)
        targets = targets.float().view(-1)

        # BCE
        bce_loss = self.bce(logits, targets)

        # Dice
        probs = torch.sigmoid(logits)

        intersection = (probs * targets).sum()

        dice_score = (2 * intersection + self.eps) / (
            probs.sum() + targets.sum() + self.eps
        )

        dice_loss = 1 - dice_score

        return self.w * bce_loss + (1 - self.w) * dice_loss
