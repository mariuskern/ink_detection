# from __future__ import annotations

# from dataclasses import dataclass
# from typing import Optional

# import torch
# import numpy as np


# @dataclass
# class CutMixResult:
#     image: torch.Tensor
#     pseudo_label: torch.Tensor
#     confidence: torch.Tensor
#     valid_mask: torch.Tensor
#     mask: torch.Tensor


# class CutMix:
#     """
#     CutMix for semi-supervised semantic segmentation.

#     The same CutMix region is applied to

#         - image
#         - pseudo labels
#         - confidence map
#         - valid mask

#     exactly as done in UniMatch.

#     Parameters
#     ----------
#     alpha:
#         Beta distribution parameter.

#     probability:
#         Probability that CutMix is applied.
#     """

#     def __init__(
#         self,
#         alpha: float = 1.0,
#         probability: float = 0.5,
#     ):
#         self.alpha = alpha
#         self.probability = probability

#     # ------------------------------------------------------------------

#     def __call__(
#         self,
#         image: torch.Tensor,
#         image_mix: torch.Tensor,
#         pseudo_label: torch.Tensor,
#         pseudo_label_mix: torch.Tensor,
#         confidence: torch.Tensor,
#         confidence_mix: torch.Tensor,
#         valid_mask: torch.Tensor,
#         valid_mask_mix: torch.Tensor,
#     ) -> CutMixResult:

#         if torch.rand(()) > self.probability:

#             return CutMixResult(
#                 image=image,
#                 pseudo_label=pseudo_label,
#                 confidence=confidence,
#                 valid_mask=valid_mask,
#                 mask=torch.zeros_like(valid_mask, dtype=torch.bool),
#             )

#         mask = self.generate_mask(
#             batch_size=image.shape[0],
#             height=image.shape[-2],
#             width=image.shape[-1],
#             device=image.device,
#         )

#         image = self.mix(image, image_mix, mask)

#         pseudo_label = self.mix(
#             pseudo_label,
#             pseudo_label_mix,
#             mask,
#         )

#         confidence = self.mix(
#             confidence,
#             confidence_mix,
#             mask,
#         )

#         valid_mask = self.mix(
#             valid_mask,
#             valid_mask_mix,
#             mask,
#         )

#         return CutMixResult(
#             image=image,
#             pseudo_label=pseudo_label,
#             confidence=confidence,
#             valid_mask=valid_mask,
#             mask=mask,
#         )

#     # ------------------------------------------------------------------

#     def generate_mask(
#         self,
#         batch_size: int,
#         height: int,
#         width: int,
#         device: Optional[torch.device] = None,
#     ) -> torch.Tensor:
#         """
#         Generate one CutMix rectangle per sample.
#         """

#         mask = torch.zeros(
#             batch_size,
#             height,
#             width,
#             dtype=torch.bool,
#             device=device,
#         )

#         for i in range(batch_size):

#             lam = np.random.beta(self.alpha, self.alpha)

#             cut_ratio = np.sqrt(1.0 - lam)

#             cut_w = int(width * cut_ratio)
#             cut_h = int(height * cut_ratio)

#             cx = np.random.randint(width)
#             cy = np.random.randint(height)

#             x1 = np.clip(cx - cut_w // 2, 0, width)
#             x2 = np.clip(cx + cut_w // 2, 0, width)

#             y1 = np.clip(cy - cut_h // 2, 0, height)
#             y2 = np.clip(cy + cut_h // 2, 0, height)

#             mask[i, y1:y2, x1:x2] = True

#         return mask

#     # ------------------------------------------------------------------

#     @staticmethod
#     def mix(
#         tensor_a: torch.Tensor,
#         tensor_b: torch.Tensor,
#         mask: torch.Tensor,
#     ) -> torch.Tensor:
#         """
#         Apply CutMix.

#         Supports

#             [B,C,H,W]
#             [B,H,W]
#             [B,1,H,W]
#         """

#         mixed = tensor_a.clone()

#         if tensor_a.ndim == 4:
#             mask = mask.unsqueeze(1)

#         mixed[mask] = tensor_b[mask]

#         return mixed