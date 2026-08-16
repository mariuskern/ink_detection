from __future__ import annotations

from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl
import numpy as np
from torchvision import tv_tensors

# from .cutmix import CutMix
from utils import calculate_dice, calculate_iou, save_predictions_as_images, PredictionMerger


class UniMatchModule(pl.LightningModule):
    """
    UniMatch-style semi-supervised segmentation module.

    The module supports:

        - supervised segmentation
        - weak-to-strong consistency
        - two strong augmentations
        - feature perturbation
        - pseudo-label confidence thresholding
        - optional CutMix
        - binary segmentation

    Expected batch format:

        {
            "tile": Tensor[B, C, H, W],
            "label": Tensor[B, H, W] or Tensor[B, 1, H, W],
            "mask": Tensor[B, H, W] or Tensor[B, 1, H, W],
            "is_pseudo": Tensor[B]
        }

    `is_pseudo == False`
        -> labeled sample

    `is_pseudo == True`
        -> unlabeled sample
    """

    def __init__(
        self,
        model,
        loss,
        lr,
        warmup_epochs,
        max_epochs,
        weight_decay,
        val_img_dir,
        val_image_every_n_epochs,
        batch_size,
        backbone_lr_multiplier: float = 1.0,
        threshold: float = 0.5,
        confidence_threshold: float = 0.95,
        lambda_u: float = 1.0,
        strong_loss_weight: float = 0.25,
        fp_loss_weight: float = 0.5,
        weak_transform: Optional[Callable] = None,
        strong_transform_1: Optional[Callable] = None,
        strong_transform_2: Optional[Callable] = None,
        # use_cutmix: bool = True,
        # cutmix_alpha: float = 1.0,
        # cutmix_probability: float = 0.5,
        # ignore_index: int = 255,
        feature_dropout_p: float = 0.5,
    ):
        super().__init__()

        self.lr = lr
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.val_img_dir = val_img_dir
        self.val_image_every_n_epochs = val_image_every_n_epochs
        self.threshold = threshold
        self.confidence_threshold = confidence_threshold
        self.backbone_lr_multiplier = backbone_lr_multiplier
        self.lambda_u = lambda_u
        self.strong_loss_weight = strong_loss_weight
        self.fp_loss_weight = fp_loss_weight
        self.weak_transform = weak_transform
        self.strong_transform_1 = strong_transform_1
        self.strong_transform_2 = strong_transform_2
        # self.ignore_index = ignore_index
        self.feature_dropout_p = feature_dropout_p
        # self.use_cutmix = use_cutmix

        self.model = model
        self.loss = loss

        # self.cutmix = CutMix(
        #     alpha=cutmix_alpha,
        #     probability=cutmix_probability,
        # )
        self.prediction_merger = PredictionMerger()

        # # BCE with per-pixel reduction.
        # #
        # # We apply the valid/confidence mask manually.
        # self.bce = nn.BCEWithLogitsLoss(
        #     reduction="none"
        # )

        self.save_hyperparameters(
            ignore=[
                "model",
                "loss",
                "weak_transform",
                "strong_transform_1",
                "strong_transform_2",
            ]
        )

    # ======================================================================
    # Forward
    # ======================================================================

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    # ======================================================================
    # Feature perturbation
    # ======================================================================

    def forward_with_feature_perturbation(self, x: torch.Tensor):
        """
        Forward pass with feature perturbation.

        Preferred model interface:

            features = model.encode(x)
            logits = model.decode(features)

        The feature tensor is then perturbed with dropout before being
        passed through the decoder.

        This corresponds conceptually to the feature perturbation used
        in the official UniMatch implementation.

        Returns
        -------
        logits:
            Normal prediction.

        logits_fp:
            Prediction from perturbed features.
        """

        # if not hasattr(self.model, "encode"):
        #     raise AttributeError("The model must implement `encode()` for feature perturbation.")

        # if not hasattr(self.model, "decode"):
        #     raise AttributeError("The model must implement `decode()` for feature perturbation.")

        # features = self.model.encode(x)
        # logits = self.model.decode(features)

        # features_fp = self._dropout_features(features)
        # logits_fp = self.model.decode(features_fp)

        # return logits, logits_fp

        features = self.model.encode(x)
        logits = self.model.decode(features)

        x_f, conv_block, enc1, enc2, enc3 = features

        features_fp = (
            self._dropout_features(x_f),
            conv_block,
            enc1,
            enc2,
            enc3,
        )

        logits_fp = self.model.decode(features_fp)

        return logits, logits_fp

    def _dropout_features(self, features):
        """
        Apply dropout to encoded features.
        """

        if isinstance(features, torch.Tensor):
            if features.ndim == 4:
                return F.dropout2d(features, p=self.feature_dropout_p, training=self.training)

            return F.dropout(features, p=self.feature_dropout_p, training=self.training)

        if isinstance(features, tuple):
            return tuple(self._dropout_features(feature) for feature in features)

        if isinstance(features, list):
            return [self._dropout_features(feature) for feature in features]

        if isinstance(features, dict):
            return {key: self._dropout_features(value) for key, value in features.items()}

        raise TypeError(f"Unsupported feature type: {type(features)}")

    @torch.no_grad()
    def generate_pseudo_labels(self, logits: torch.Tensor):
        """
        Generate pseudo labels from weak predictions.

        For binary segmentation:

            p = sigmoid(logits)

            pseudo_label = p >= 0.5

            confidence = max(p, 1-p)

        Therefore confidence represents how certain the model is about
        either foreground or background.
        """

        # if logits.ndim != 4:
        #     raise ValueError(f"Expected logits with shape [B, 1, H, W], got {tuple(logits.shape)}")

        # if logits.shape[1] != 1:
        #     raise ValueError(f"This UniMatch module is configured for binarysegmentation, but received {logits.shape[1]} channels.")

        # probability = torch.sigmoid(logits[:, 0])
        # pseudo_label = (probability >= self.threshold).float()
        # confidence = torch.maximum(probability, 1.0 - probability,)

        # valid_mask = torch.ones_like(confidence, dtype=torch.bool)

        # pseudo_label = (logits >= self.threshold).astype(np.uint8)
        # confidence = np.maximum(logits, 1 - logits)
        # mask = np.ones_like(pseudo_label, dtype=bool)
        prob = torch.sigmoid(logits)
        pseudo_label = (prob >= self.threshold).to(torch.uint8)
        confidence = torch.maximum(prob, 1 - prob)
        mask = torch.ones_like(pseudo_label, dtype=torch.bool)

        return pseudo_label, confidence, mask

    # def supervised_loss(self, logits: torch.Tensor, target: torch.Tensor, valid_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    #     """
    #     BCE loss for labeled samples.

    #     Ignore pixels with `ignore_index` or where valid_mask == 0.
    #     """

    #     target = self._prepare_target(target)

    #     logits = self._prepare_logits(logits)

    #     valid = target != self.ignore_index

    #     if valid_mask is not None:
    #         valid = valid & self._prepare_valid_mask(valid_mask)

    #     # BCEWithLogitsLoss expects values in [0, 1].
    #     #
    #     # Ignored pixels are replaced by 0 because they are removed
    #     # from the loss immediately afterwards.
    #     safe_target = target.float().clamp(
    #         0.0,
    #         1.0,
    #     )

    #     loss = self.bce(
    #         logits,
    #         safe_target,
    #     )

    #     loss = loss * valid

    #     denominator = valid.sum().clamp_min(1)

    #     return loss.sum() / denominator

    # ======================================================================
    # Consistency loss
    # ======================================================================

    # def consistency_loss(self, logits: torch.Tensor, pseudo_label: torch.Tensor, confidence: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    #     """
    #     Consistency loss against pseudo labels.

    #     A pixel contributes to the loss only if:

    #         confidence >= conf_thresh

    #     and

    #         valid_mask == True

    #     This is the binary equivalent of UniMatch's pixel-wise
    #     CrossEntropy loss with confidence masking.
    #     """

    #     # logits = self._prepare_logits(logits)
    #     # pseudo_label = self._prepare_target(pseudo_label)
    #     # loss = self.bce(logits, pseudo_label.float())
    #     # confidence_mask = (confidence >= self.conf_thresh)
    #     # mask = confidence_mask & valid_mask

    #     # loss = loss * mask

    #     # # Same normalization idea as the official implementation:
    #     # #
    #     # # loss.sum() / number of valid pixels
    #     # #
    #     # # rather than dividing by the number of confident pixels.
    #     # denominator = valid_mask.sum().clamp_min(1)

    #     # return loss.sum() / denominator

    #     mask = valid_mask * (confidence > self.confidence_threshold)
    #     loss = self.loss(logits, pseudo_label, mask)
    #     return loss

    # ======================================================================
    # Helpers
    # ======================================================================

    # @staticmethod
    # def _prepare_valid_mask(
    #     valid_mask: torch.Tensor,
    # ) -> torch.Tensor:
    #     """
    #     Convert a pixel validity mask to [B,H,W] bool.
    #     """

    #     if valid_mask.ndim == 4:

    #         if valid_mask.shape[1] != 1:
    #             raise ValueError(
    #                 "Expected valid_mask with shape [B,1,H,W], "
    #                 f"got {tuple(valid_mask.shape)}"
    #             )

    #         valid_mask = valid_mask[:, 0]

    #     if valid_mask.ndim != 3:
    #         raise ValueError(
    #             "Expected valid_mask with shape [B,H,W], "
    #             f"got {tuple(valid_mask.shape)}"
    #         )

    #     return valid_mask > 0.5

    # @staticmethod
    # def _prepare_target(
    #     target: torch.Tensor,
    # ) -> torch.Tensor:
    #     """
    #     Convert segmentation target to [B,H,W].
    #     """

    #     if target.ndim == 4:

    #         if target.shape[1] != 1:
    #             raise ValueError(
    #                 "Expected target with shape [B,1,H,W], "
    #                 f"got {tuple(target.shape)}"
    #             )

    #         target = target[:, 0]

    #     if target.ndim != 3:
    #         raise ValueError(
    #             "Expected target with shape [B,H,W], "
    #             f"got {tuple(target.shape)}"
    #         )

    #     return target

    # ======================================================================

    # @staticmethod
    # def _prepare_logits(
    #     logits: torch.Tensor,
    # ) -> torch.Tensor:
    #     """
    #     Convert binary logits to [B,H,W].
    #     """

    #     if logits.ndim != 4:
    #         raise ValueError(
    #             "Expected logits with shape [B,1,H,W], "
    #             f"got {tuple(logits.shape)}"
    #         )

    #     if logits.shape[1] != 1:
    #         raise ValueError(
    #             "Expected exactly one output channel for binary "
    #             f"segmentation, got {logits.shape[1]}."
    #         )

    #     return logits[:, 0]

    # ======================================================================
    # Augmentation helper
    # ======================================================================

    # @staticmethod
    # def _apply_transform(
    #     images: torch.Tensor,
    #     transform: Optional[Callable],
    # ) -> torch.Tensor:
    #     """
    #     Apply an image transform independently to every sample.

    #     The transform receives:

    #         [C,H,W]

    #     and must return:

    #         [C,H,W]
    #     """

    #     if transform is None:
    #         return images

    #     return torch.stack(
    #         [
    #             transform(image)
    #             for image in images
    #         ],
    #         dim=0,
    #     )

    # ======================================================================
    # Training step
    # ======================================================================

    def training_step(self, batch, batch_idx):
        """
        UniMatch training step.

        Expected batch:

            {
                "tile": Tensor[B,C,H,W],
                "label": Tensor[B,H,W] or Tensor[B,1,H,W],
                "mask": Tensor[B,H,W] or Tensor[B,1,H,W],
                "is_pseudo": Tensor[B]
            }

        Labeled samples:
            is_pseudo == False

        Unlabeled samples:
            is_pseudo == True

        The unlabeled part is split into two groups:

            group A -> normal unlabeled samples
            group B -> CutMix source samples

        This allows CutMix without requiring a second DataLoader.
        """

        tile = batch["tile"]
        
        label = batch["label"]
        mask = batch["mask"]
        has_label = batch["has_label"]
        has_mask = batch["has_mask"]

        tile_l = tile[has_label]
        tile_u = tile[~has_label]
        label_l = label[has_label]
        # label_u = label[~has_label]
        mask_l = mask[has_label]
        mask_u = mask[~has_label]

        # ==================================================================
        # 1. Supervised loss
        # ==================================================================

        if tile_l.shape[0] > 0: # has_label.any():
            logits_l = self(tile_l)
            # loss_l = self.supervised_loss(logits_l, label_l, mask_l)
            loss_l = self.loss(logits_l, label_l, mask_l)
        else:
            # loss_l = torch.tensor(0.0, device=self.device)
            loss_l = tile.sum() * 0.0

        # ==================================================================
        # No unlabeled samples
        # ==================================================================

        if tile_u.shape[0] == 0:
            loss = loss_l
            self._log_supervised_only(loss=loss, loss_l=loss_l, batch_size=tile.shape[0])
            return loss

        # ==================================================================
        # 2. Create weak and strong views
        # ==================================================================

        # tile_u_w = self._apply_transform(tile_u, self.weak_transform)
        # tile_u_s1 = self._apply_transform(tile_u, self.strong_transform_1)
        # tile_u_s2 = self._apply_transform(tile_u, self.strong_transform_2)

        tile_u = tv_tensors.Image(tile_u)
        tile_u_w = self.weak_transform(tile_u)
        tile_u_w = tile_u_w.as_subclass(torch.Tensor)

        # ==================================================================
        # 3. Generate pseudo labels from weak prediction
        # ==================================================================

        with torch.no_grad():
            self.model.eval()

            pred_u_w = self(tile_u_w)

            pseudo_label, confidence, valid_mask = self.generate_pseudo_labels(pred_u_w)

            if mask_u is not None:
                # valid_mask = valid_mask & self._prepare_valid_mask(mask_u)
                valid_mask = valid_mask * mask_u

            # valid_mask &= confidence > self.confidence_threshold
            valid_mask = valid_mask * (confidence > self.confidence_threshold)

            self.model.train()

        # tile_u = tv_tensors.Image(tile_u)
        pseudo_label = tv_tensors.Mask(pseudo_label)
        valid_mask = tv_tensors.Mask(valid_mask)

        tile_u_s1, label_u_s1, mask_u_s1 = self.strong_transform_1(tile_u, pseudo_label, valid_mask)
        tile_u_s2, label_u_s2, mask_u_s2 = self.strong_transform_2(tile_u, pseudo_label, valid_mask)

        tile_u_s1 = tile_u_s1.as_subclass(torch.Tensor)
        label_u_s1 = label_u_s1.as_subclass(torch.Tensor)
        mask_u_s1 = mask_u_s1.as_subclass(torch.Tensor)
        tile_u_s2 = tile_u_s2.as_subclass(torch.Tensor)
        label_u_s2 = label_u_s2.as_subclass(torch.Tensor)
        mask_u_s2 = mask_u_s2.as_subclass(torch.Tensor)

        # # ==================================================================
        # # 4. Prepare a second unlabeled group for CutMix
        # # ==================================================================

        # if self.use_cutmix and tile_u.shape[0] > 1:

        #     (
        #         tile_u_mix,
        #         pseudo_mix,
        #         confidence_mix,
        #         valid_mix,
        #     ) = self._create_cutmix_source(
        #         tile_u=tile_u,
        #         tile_u_w=tile_u_w,
        #         pseudo_label=pseudo_label,
        #         confidence=confidence,
        #         valid_mask=valid_mask,
        #     )

        # else:

        #     tile_u_mix = tile_u
        #     pseudo_mix = pseudo_label
        #     confidence_mix = confidence
        #     valid_mix = valid_mask

        # # ==================================================================
        # # 5. Apply CutMix to Strong View 1
        # # ==================================================================

        # if self.use_cutmix and tile_u.shape[0] > 1:

        #     result_1 = self.cutmix(
        #         image=tile_u_s1,
        #         image_mix=self._apply_transform(
        #             tile_u_mix,
        #             self.strong_transform_1,
        #         ),
        #         pseudo_label=pseudo_label,
        #         pseudo_label_mix=pseudo_mix,
        #         confidence=confidence,
        #         confidence_mix=confidence_mix,
        #         valid_mask=valid_mask,
        #         valid_mask_mix=valid_mix,
        #     )

        #     tile_u_s1 = result_1.image
        #     pseudo_s1 = result_1.pseudo_label
        #     confidence_s1 = result_1.confidence
        #     valid_s1 = result_1.valid_mask

        # else:

        #     pseudo_s1 = pseudo_label
        #     confidence_s1 = confidence
        #     valid_s1 = valid_mask

        # # ==================================================================
        # # 6. Apply CutMix to Strong View 2
        # # ==================================================================

        # if self.use_cutmix and tile_u.shape[0] > 1:

        #     result_2 = self.cutmix(
        #         image=tile_u_s2,
        #         image_mix=self._apply_transform(
        #             tile_u_mix,
        #             self.strong_transform_2,
        #         ),
        #         pseudo_label=pseudo_label,
        #         pseudo_label_mix=pseudo_mix,
        #         confidence=confidence,
        #         confidence_mix=confidence_mix,
        #         valid_mask=valid_mask,
        #         valid_mask_mix=valid_mix,
        #     )

        #     tile_u_s2 = result_2.image
        #     pseudo_s2 = result_2.pseudo_label
        #     confidence_s2 = result_2.confidence
        #     valid_s2 = result_2.valid_mask

        # else:

        #     pseudo_s2 = pseudo_label
        #     confidence_s2 = confidence
        #     valid_s2 = valid_mask

        # ==================================================================
        # 7. Strong predictions
        # ==================================================================

        pred_u_s1 = self(tile_u_s1)
        pred_u_s2 = self(tile_u_s2)

        # ==================================================================
        # 8. Feature perturbation
        # ==================================================================

        _, pred_u_w_fp = (self.forward_with_feature_perturbation(tile_u_w))

        # ==================================================================
        # 9. Strong consistency losses
        # ==================================================================

        # loss_u_s1 = self.consistency_loss(logits=pred_u_s1, pseudo_label=pseudo_s1, confidence=confidence_s1, valid_mask=valid_s1)
        # loss_u_s2 = self.consistency_loss(logits=pred_u_s2, pseudo_label=pseudo_s2, confidence=confidence_s2, valid_mask=valid_s2)
        loss_u_s1 = self.loss(pred_u_s1, label_u_s1, mask_u_s1)
        loss_u_s2 = self.loss(pred_u_s2, label_u_s2, mask_u_s2)

        # ==================================================================
        # 10. Feature perturbation consistency
        # ==================================================================

        # loss_u_fp = self.consistency_loss(logits=pred_u_w_fp, pseudo_label=pseudo_label, confidence=confidence, valid_mask=valid_mask)
        loss_u_fp = self.loss(pred_u_w_fp, pseudo_label, valid_mask)

        # ==================================================================
        # 11. UniMatch unsupervised loss
        # ==================================================================

        loss_u = self.strong_loss_weight * loss_u_s1 + self.strong_loss_weight * loss_u_s2 + self.fp_loss_weight * loss_u_fp

        # ==================================================================
        # 12. Total loss
        # ==================================================================

        loss = loss_l + self.lambda_u * loss_u

        # ==================================================================
        # 13. Logging
        # ==================================================================

        # pseudo_ratio = ((confidence >= self.confidence_threshold) & valid_mask).float().mean()
        pseudo_ratio = valid_mask.float().mean()

        # self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=tile.shape[0])
        # self.log("train/loss_labeled",loss_l, on_step=True, on_epoch=True, batch_size=tile.shape[0])
        # self.log("train/loss_unlabeled", loss_u, on_step=True, on_epoch=True, batch_size=tile.shape[0])
        # self.log("train/loss_strong_1", loss_u_s1, on_step=True, on_epoch=True, batch_size=tile.shape[0])
        # self.log("train/loss_strong_2", loss_u_s2, on_step=True, on_epoch=True, batch_size=tile.shape[0])
        # self.log( "train/loss_feature_fp", loss_u_fp, on_step=True, on_epoch=True, batch_size=tile.shape[0])
        # self.log("train/pseudo_ratio", pseudo_ratio, prog_bar=True, on_step=True, on_epoch=True, batch_size=tile.shape[0])

        self.log_dict(
            {
                "train/loss": loss,
                "train/loss_labeled": loss_l,
                "train/loss_unlabeled": loss_u,
                "train/loss_strong_1": loss_u_s1,
                "train/loss_strong_2": loss_u_s2,
                "train/loss_feature_fp": loss_u_fp,
                "train/pseudo_ratio": pseudo_ratio,
            },
            on_step=True,
            on_epoch=True,
            batch_size=tile.shape[0],
        )

        return loss

    # ======================================================================
    # CutMix source
    # ======================================================================

    # @torch.no_grad()
    # def _create_cutmix_source(
    #     self,
    #     tile_u: torch.Tensor,
    #     tile_u_w: torch.Tensor,
    #     pseudo_label: torch.Tensor,
    #     confidence: torch.Tensor,
    #     valid_mask: torch.Tensor,
    # ):
    #     """
    #     Create a second unlabeled source by cyclically shifting the batch.

    #     Example:

    #         original:
    #             [A, B, C, D]

    #         source:
    #             [B, C, D, A]

    #     This gives every sample a different CutMix partner without
    #     requiring another DataLoader.
    #     """

    #     permutation = torch.roll(
    #         torch.arange(
    #             tile_u.shape[0],
    #             device=tile_u.device,
    #         ),
    #         shifts=-1,
    #     )

    #     tile_u_mix = tile_u[
    #         permutation
    #     ]

    #     tile_u_w_mix = tile_u_w[
    #         permutation
    #     ]

    #     pseudo_mix = pseudo_label[
    #         permutation
    #     ]

    #     confidence_mix = confidence[
    #         permutation
    #     ]

    #     valid_mix = valid_mask[
    #         permutation
    #     ]

    #     return (
    #         tile_u_mix,
    #         pseudo_mix,
    #         confidence_mix,
    #         valid_mix,
    #     )

    # ======================================================================
    # Supervised-only logging
    # ======================================================================

    def _log_supervised_only(self, loss: torch.Tensor, loss_l: torch.Tensor, batch_size: int):
        self.log_dict(
            {
                "train/loss": loss,
                "train/loss_labeled": loss_l,
                "train/loss_unlabeled": 0.0,
                "train/pseudo_ratio": 0.0,
            },
            on_step=True,
            on_epoch=True,
            batch_size=batch_size,
        )

    # ======================================================================
    # Optimizer
    # ======================================================================

    def configure_optimizers(self):
        """
        SGD optimizer + Poly learning-rate schedule.

        This follows the learning-rate strategy used by UniMatch:

            lr = lr_0 * (1 - iter / total_iters) ** 0.9

        If the model has a `backbone` attribute, its parameters use the
        base learning rate while all remaining parameters use:

            lr * backbone_lr_multiplier
        """

        if hasattr(self.model, "backbone"):
            backbone_params = list(self.model.backbone.parameters())

            backbone_ids = {id(parameter) for parameter in backbone_params}

            head_params = [parameter for parameter in self.model.parameters() if id(parameter) not in backbone_ids]

            optimizer = torch.optim.SGD(
                [
                    {
                        "params": backbone_params,
                        "lr": self.lr,
                    },
                    {
                        "params": head_params,
                        "lr": self.lr * self.backbone_lr_multiplier,
                    },
                ],
                momentum=0.9,
                weight_decay=self.weight_decay,
            )
        else:
            optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=self.lr,
                momentum=0.9,
                weight_decay=self.weight_decay,
            )

        # --------------------------------------------------------------
        # Poly learning rate
        # --------------------------------------------------------------

        def poly_lr(step: int):
            total_steps = self.trainer.estimated_stepping_batches

            if total_steps <= 1:
                return 1.0

            progress = min(step / (total_steps - 1), 1.0)

            return (1.0 - progress) ** 0.9

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=poly_lr)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

    # ======================================================================
    # Validation
    # ======================================================================

    def validation_step(self, batch, batch_idx):
        """
        Standard supervised validation.

        Validation does NOT use pseudo labels or UniMatch augmentations.
        """

        tile = batch["tile"]
        label = batch["label"]
        mask = batch["mask"]
        has_label = batch["has_label"]
        segment = batch["segment"]
        x = batch["x"]
        y = batch["y"]
        shape = batch["shape"]

        if not has_label.any():
            return None

        tile = tile[has_label]
        label = label[has_label]
        mask = mask[has_label]
        # segment = segment[has_label]
        segment = [s for s, keep in zip(segment, has_label.cpu().tolist()) if keep]
        x = x[has_label]
        y = y[has_label]
        shape = shape[has_label]

        logits = self(tile)

        # loss = self.supervised_loss(logits, label, mask)
        loss = self.loss(logits, label, mask)

        # # prediction = torch.sigmoid(logits[:, 0]) >= self.threshold
        # # target = self._prepare_target(label)
        # target = label

        # # valid = (target != self.ignore_index) & self._prepare_valid_mask(mask)
        # valid = target * mask

        # target_binary = target > 0.5
        # target_binary = target_binary * valid

        # # prediction = prediction & valid

        # # dice = self._dice_score(prediction, target_binary, valid)
        # # iou = self._iou_score(prediction, target_binary, valid)

        dice = calculate_dice(logits, label, mask)
        iou = calculate_iou(logits, label, mask)

        if self._export_validation_predictions_this_epoch():
            probs = torch.sigmoid(logits).squeeze(1).detach().float().cpu().numpy()
            targets = label.squeeze(1).detach().float().cpu().numpy()
            valid_masks = mask.squeeze(1).detach().float().cpu().numpy()

            for i in range(len(segment)):
                self.prediction_merger.add(
                    segment=segment[i],
                    x=int(x[i]),
                    y=int(y[i]),
                    prediction=probs[i],
                    shape=shape[i].tolist(),
                    target=targets[i],
                    valid_mask=valid_masks[i],
                )

        self.log_dict(
            {
                "val/loss": loss,
                "val/dice": dice,
                "val/iou": iou,
                "train/pseudo_ratio": 0.0,
            },
            on_step=True,
            on_epoch=True,
            batch_size=tile.shape[0],
        )

        return {
            "val_loss": loss.detach(),
            "dice": dice.detach(),
            "iou": iou.detach(),
        }

    # ======================================================================
    # Dice
    # ======================================================================

    # @staticmethod
    # def _dice_score(
    #     prediction: torch.Tensor,
    #     target: torch.Tensor,
    #     valid: torch.Tensor,
    #     eps: float = 1e-7,
    # ) -> torch.Tensor:
    #     """
    #     Binary Dice score.

    #     Dice = 2 * intersection / (prediction + target)
    #     """

    #     prediction = prediction & valid
    #     target = target & valid

    #     prediction = prediction.float()
    #     target = target.float()

    #     intersection = (
    #         prediction * target
    #     ).sum()

    #     denominator = (
    #         prediction.sum()
    #         + target.sum()
    #     )

    #     return (
    #         2.0 * intersection + eps
    #     ) / (
    #         denominator + eps
    #     )

    # ======================================================================
    # IoU
    # ======================================================================

    # @staticmethod
    # def _iou_score(
    #     prediction: torch.Tensor,
    #     target: torch.Tensor,
    #     valid: torch.Tensor,
    #     eps: float = 1e-7,
    # ) -> torch.Tensor:
    #     """
    #     Binary Intersection over Union.

    #     IoU = intersection / union
    #     """

    #     prediction = prediction & valid
    #     target = target & valid

    #     prediction = prediction.float()
    #     target = target.float()

    #     intersection = (
    #         prediction * target
    #     ).sum()

    #     union = (
    #         prediction
    #         + target
    #         - prediction * target
    #     ).sum()

    #     return (
    #         intersection + eps
    #     ) / (
    #         union + eps
    #     )

    # ======================================================================
    # Test
    # ======================================================================

    # def test_step(
    #     self,
    #     batch,
    #     batch_idx,
    # ):
    #     """
    #     Test step using the same metrics as validation.
    #     """

    #     tile = batch["tile"]
    #     label = batch["label"]
    #     valid_mask_batch = batch["mask"]

    #     logits = self.model(tile)

    #     loss = self.supervised_loss(
    #         logits,
    #         label,
    #         valid_mask_batch,
    #     )

    #     prediction = (
    #         torch.sigmoid(logits[:, 0])
    #         >= self.threshold
    #     )

    #     target = self._prepare_target(
    #         label
    #     )

    #     valid = (
    #         target != self.ignore_index
    #     ) & self._prepare_valid_mask(valid_mask_batch)

    #     target_binary = (
    #         target > 0.5
    #     )

    #     dice = self._dice_score(
    #         prediction,
    #         target_binary,
    #         valid,
    #     )

    #     iou = self._iou_score(
    #         prediction,
    #         target_binary,
    #         valid,
    #     )

    #     self.log(
    #         "test/loss",
    #         loss,
    #         on_step=False,
    #         on_epoch=True,
    #         batch_size=tile.shape[0],
    #     )

    #     self.log(
    #         "test/dice",
    #         dice,
    #         on_step=False,
    #         on_epoch=True,
    #     )

    #     self.log(
    #         "test/iou",
    #         iou,
    #         on_step=False,
    #         on_epoch=True,
    #     )

    #     return {
    #         "loss": loss.detach(),
    #         "dice": dice.detach(),
    #         "iou": iou.detach(),
    #     }

    # ======================================================================
    # Predict
    # ======================================================================

    # def predict_step(
    #     self,
    #     batch,
    #     batch_idx,
    #     dataloader_idx=0,
    # ):
    #     """
    #     Prediction interface for Lightning Trainer.predict().

    #     Returns probabilities and binary predictions.
    #     """

    #     tile = batch["tile"]

    #     logits = self.model(tile)

    #     probability = torch.sigmoid(
    #         logits
    #     )

    #     prediction = (
    #         probability >= self.threshold
    #     )

    #     return {
    #         "logits": logits,
    #         "probability": probability,
    #         "prediction": prediction,
    #     }

    # ======================================================================
    # Epoch hooks
    # ======================================================================

    # def on_train_epoch_end(self):
    #     """
    #     Log the current learning rate.
    #     """

    #     optimizer = self.optimizers()

    #     if isinstance(
    #         optimizer,
    #         (list, tuple),
    #     ):
    #         optimizer = optimizer[0]

    #     lr = optimizer.param_groups[0]["lr"]

    #     self.log(
    #         "train/lr",
    #         lr,
    #         prog_bar=False,
    #         on_step=False,
    #         on_epoch=True,
    #     )

    def on_validation_epoch_end(self):
        if self.global_rank != 0:
            self.prediction_merger.clear()
            return

        if self._export_validation_predictions_this_epoch():
            metrics = self.prediction_merger.get_binary_metrics()
            if metrics is not None:
                self.log("val/segment_dice", metrics["dice"], prog_bar=True)
                self.log("val/segment_iou", metrics["iou"])

            results = self.prediction_merger.get_predictions()
            save_predictions_as_images(self.val_img_dir, results, save_probability=True, name_suffix=f"epoch_{self.current_epoch:04d}", create_subfolders=False)

        self.prediction_merger.clear()

    def _export_validation_predictions_this_epoch(self):
        return self.current_epoch % self.val_image_every_n_epochs == 0
