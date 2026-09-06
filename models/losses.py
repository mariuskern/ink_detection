# import torch
# import torch.nn as nn


# def build_loss(cfg):
#     match cfg.loss.name:
#         case "dice_bce":
#             return DiceBCELoss(w=cfg.loss.w, eps=cfg.loss.eps)


# class DiceBCELoss(nn.Module):
#     def __init__(self, w=0.5, eps=1e-6):
#         super().__init__()

#         self.w = w
#         self.eps = eps

#         self.bce = nn.BCEWithLogitsLoss(reduction="none")

#     def forward(self, logits, targets, mask=None):
#         """
#         Args:
#             logits: Modelloutput, Shape (B,) oder (B,1)
#             targets: Labels, Shape (B,) oder (B,1), Werte 0/1
#         """

#         targets = targets.float()

#         if logits.shape != targets.shape:
#             raise ValueError(f"logits and targets must have the same shape, got {tuple(logits.shape)} and {tuple(targets.shape)}")

#         if mask is None:
#             valid = torch.ones_like(targets)
#         else:
#             if mask.shape != targets.shape:
#                 raise ValueError(f"mask and targets must have the same shape, got {tuple(mask.shape)} and {tuple(targets.shape)}")
#             valid = mask.to(dtype=targets.dtype, device=targets.device)

#         # BCE
#         bce_loss = self.bce(logits, targets)
#         bce_loss = (bce_loss * valid).sum() / valid.sum().clamp(min=1)

#         # Dice
#         probs = torch.sigmoid(logits)

#         probs = probs * valid
#         targets = targets * valid

#         # intersection = (probs * targets).sum()
#         # dice_score = (2 * intersection + self.eps) / (probs.sum() + targets.sum() + self.eps)

#         dims = tuple(range(1, logits.ndim))
#         intersection = (probs * targets).sum(dim=dims)
#         union = probs.sum(dim=dims) + targets.sum(dim=dims)
#         dice = (2 * intersection + self.eps) / (union + self.eps)
#         dice_score = dice.mean()

#         dice_loss = 1 - dice_score

#         return self.w * bce_loss + (1 - self.w) * dice_loss



# First loss with mask


# import torch
# import torch.nn as nn


# def build_loss(cfg):
#     match cfg.loss.name:
#         case "dice_bce":
#             return DiceBCELoss(w=cfg.loss.w, eps=cfg.loss.eps)


# class DiceBCELoss(nn.Module):
#     def __init__(self, w=0.5, eps=1e-6):
#         super().__init__()

#         self.w = w
#         self.eps = eps

#         self.bce = nn.BCEWithLogitsLoss(reduction="none")

#     def forward(self, logits, targets, mask=None):
#         """
#         Args:
#             logits: Modelloutput, Shape (B,) oder (B,1)
#             targets: Labels, Shape (B,) oder (B,1), Werte 0/1
#         """

#         # gleiche Shape erzwingen
#         # logits = logits.view(-1)
#         # targets = targets.float().view(-1)
#         targets = targets.float()

#         # BCE
#         bce_loss = self.bce(logits, targets)

#         # Dice
#         probs = torch.sigmoid(logits)

#         if mask is not None:
#             mask = mask.float()
#             bce_loss = bce_loss * mask
#             probs = probs * mask
#             targets = targets * mask

#         if mask is not None:
#             bce_loss = bce_loss.sum() / mask.sum().clamp(min=1)
#         else:
#             bce_loss = bce_loss.mean()

#         intersection = (probs * targets).sum()

#         dice_score = (2 * intersection + self.eps) / (probs.sum() + targets.sum() + self.eps)

#         dice_loss = 1 - dice_score

#         return self.w * bce_loss + (1 - self.w) * dice_loss



# Original loss


# import torch
# import torch.nn as nn


# def build_loss(cfg):
#     match cfg.loss.name:
#         case "dice_bce":
#             return DiceBCELoss(w=cfg.loss.w, eps=cfg.loss.eps)


# class DiceBCELoss(nn.Module):
#     def __init__(self, w=0.5, eps=1e-6):
#         super().__init__()

#         self.w = w
#         self.eps = eps

#         self.bce = nn.BCEWithLogitsLoss()

#     def forward(self, logits, targets, mask=None):
#         """
#         Args:
#             logits: Modelloutput, Shape (B,) oder (B,1)
#             targets: Labels, Shape (B,) oder (B,1), Werte 0/1
#         """

#         # gleiche Shape erzwingen
#         logits = logits.view(-1)
#         targets = targets.float().view(-1)

#         # BCE
#         bce_loss = self.bce(logits, targets)

#         # Dice
#         probs = torch.sigmoid(logits)

#         intersection = (probs * targets).sum()

#         dice_score = (2 * intersection + self.eps) / (
#             probs.sum() + targets.sum() + self.eps
#         )

#         dice_loss = 1 - dice_score

#         return self.w * bce_loss + (1 - self.w) * dice_loss
