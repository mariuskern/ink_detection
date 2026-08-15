import lightning as pl
import torch

from utils import PredictionMerger, save_predictions_as_images, calculate_dice, calculate_iou


class SupervisedModule(pl.LightningModule):
    def __init__(self, model, loss, lr, warmup_epochs, max_epochs, weight_decay, val_img_dir, val_image_every_n_epochs, batch_size, iteration=None, pseudo_loss_weight=0.0, pseudo_loss_rampup_epochs=0, threshold=0.5):
        super().__init__()

        self.lr = lr
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.iteration = iteration
        self.val_img_dir = val_img_dir
        self.val_image_every_n_epochs = val_image_every_n_epochs
        self.pseudo_loss_weight = pseudo_loss_weight
        self.pseudo_loss_rampup_epochs = pseudo_loss_rampup_epochs
        self.threshold = threshold

        self.model = model
        self.loss = loss

        self.prediction_merger = PredictionMerger()

        self.save_hyperparameters(ignore=["model", "loss"])

    def forward(self, x):
        return self.model(x)

    # def transfer_batch_to_device(self, batch, device, dataloader_idx):
    #     """Keep metadata on CPU and make pinned-memory transfers asynchronous."""
    #     for key in ("tile", "label", "mask"):
    #         if batch[key] is not None:
    #             batch[key] = batch[key].to(device, non_blocking=True)
    #     return batch

    def training_step(self, batch, batch_idx):
        tile = batch["tile"]

        label = batch["label"]
        pseudo_label = batch["pseudo_label"]
        mask = batch["mask"]
        pseudo_mask = batch["pseudo_mask"]

        has_label = batch["has_label"]
        has_pseudo_label = batch["has_pseudo_label"]
        # has_mask = batch["has_mask"]
        # has_pseudo_mask = batch["has_pseudo_mask"]

        output = self(tile)

        pseudo_mask = mask * pseudo_mask

        labeled_loss = self.loss(output[has_label], label[has_label], mask[has_label]) if has_label.any() else torch.tensor(0.0, device=self.device)
        pseudo_loss = self.loss(output[has_pseudo_label], pseudo_label[has_pseudo_label], pseudo_mask[has_pseudo_label]) if has_pseudo_label.any() else torch.tensor(0.0, device=self.device)

        pseudo_weight = self._pseudo_loss_weight()

        loss = labeled_loss + pseudo_weight * pseudo_loss

        self._log_metric("train", "loss", loss, batch_size=self.batch_size, on_step=True, on_epoch=True, prog_bar=True)
        self._log_metric("train", "labeled_loss", labeled_loss, batch_size=self.batch_size, on_step=True, on_epoch=True, prog_bar=True)
        self._log_metric("train", "pseudo_loss", pseudo_loss, batch_size=self.batch_size, on_step=True, on_epoch=True, prog_bar=True)
        self._log_metric("train", "pseudo_weight", pseudo_weight, batch_size=self.batch_size, on_step=True, on_epoch=True, prog_bar=True)

        if has_label.any():
            num_labeled = has_label.sum().item()

            self._log_tile_dice("train", "tile_dice", output[has_label], label[has_label], mask[has_label], num_labeled)
            self._log_tile_iou("train", "tile_iou", output[has_label], label[has_label], mask[has_label], num_labeled)

        if has_pseudo_label.any():
            num_pseudo = has_pseudo_label.sum().item()

            self._log_tile_dice("train", "pseudo_tile_dice", output[has_pseudo_label], pseudo_label[has_pseudo_label], pseudo_mask[has_pseudo_label], num_pseudo)
            self._log_tile_iou("train", "pseudo_tile_iou", output[has_pseudo_label], pseudo_label[has_pseudo_label], pseudo_mask[has_pseudo_label], num_pseudo)

        logits_combined = torch.cat([output[has_label], output[has_pseudo_label]])
        targets_combined = torch.cat([label[has_label], pseudo_label[has_pseudo_label]])
        masks_combined = torch.cat([mask[has_label], pseudo_mask[has_pseudo_label]])

        self._log_tile_dice("train", "combined_tile_dice", logits_combined, targets_combined, masks_combined, self.batch_size)
        self._log_tile_iou("train", "combined_tile_iou", logits_combined, targets_combined, masks_combined, self.batch_size)
            
        return loss

    def validation_step(self, batch, batch_idx):
        tile = batch["tile"]
        label = batch["label"]
        mask = batch["mask"]
        has_label = batch["has_label"]
        segment = batch["segment"]
        x = batch["x"]
        y = batch["y"]
        shape = batch["shape"]

        output = self(tile)

        if has_label.any():
            loss = self.loss(output[has_label], label[has_label], mask[has_label]) if has_label.any() else torch.tensor(0.0, device=self.device)
            self._log_tile_dice("val", "tile_dice", output[has_label], label[has_label], mask[has_label], has_label.sum().item())
            self._log_tile_iou("val", "tile_iou", output[has_label], label[has_label], mask[has_label], has_label.sum().item())
        else:
            loss = output.sum() * 0.0

        self._log_metric("val", "loss", loss, batch_size=self.batch_size, on_step=True, on_epoch=True, prog_bar=True)

        if self._export_validation_predictions_this_epoch():
            probs = torch.sigmoid(output).squeeze(1).detach().float().cpu().numpy()
            targets = label.squeeze(1).detach().float().cpu().numpy()
            valid_masks = mask.squeeze(1).detach().float().cpu().numpy()

            has_label_cpu = has_label.cpu().numpy()

            for i in range(len(segment)):
                if not has_label_cpu[i]:
                    continue

                self.prediction_merger.add(
                    segment=segment[i],
                    x=int(x[i]),
                    y=int(y[i]),
                    prediction=probs[i],
                    shape=shape[i].tolist(),
                    target=targets[i],
                    valid_mask=valid_masks[i],
                )

        return loss

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

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        total_steps = self.trainer.estimated_stepping_batches
        steps_per_epoch = max(total_steps // self.max_epochs, 1)
        requested_warmup_steps = self.warmup_epochs * steps_per_epoch
        # A warmup cannot consume the entire run; otherwise the subsequent
        # cosine scheduler receives a non-positive T_max (e.g. debug runs).
        warmup_steps = min(requested_warmup_steps, max(total_steps - 1, 0))

        if warmup_steps == 0:
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": torch.optim.lr_scheduler.CosineAnnealingLR(
                        optimizer, T_max=max(total_steps, 1), eta_min=1e-6
                    ),
                    "interval": "step",
                },
            }

        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.01,  # startet bei 1% der LR
            end_factor=1.0,
            total_iters=warmup_steps, # self.warmup_epochs
        )

        # cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.max_epochs - self.warmup_epochs, eta_min=1e-6)
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(total_steps - warmup_steps, 1), eta_min=1e-6
        )

        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_steps], # [self.warmup_epochs]
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }

    def _log_tile_dice(self, stage, name, logits, target, valid_mask, batch_size):
        # prediction = (torch.sigmoid(logits) >= 0.5).to(target.dtype) * valid_mask
        # target = target * valid_mask
        # intersection = (prediction * target).sum()
        # dice = (2 * intersection + 1e-6) / (prediction.sum() + target.sum() + 1e-6)

        dice = calculate_dice(logits, target, valid_mask, threshold=self.threshold)

        # self.log(f"{stage}/tile_dice", dice, batch_size=batch_size, on_step=False, on_epoch=True, prog_bar=(stage == "val"))
        self._log_metric(stage, name, dice, batch_size=batch_size, on_step=True, on_epoch=True, prog_bar=(stage == "val"))

    def _log_tile_iou(self, stage, name, logits, target, valid_mask, batch_size):
        # prediction = (torch.sigmoid(logits) >= 0.5).to(target.dtype) * valid_mask
        # target = target * valid_mask

        # intersection = (prediction * target).sum()
        # union = prediction.sum() + target.sum() - intersection

        # iou = (intersection + 1e-6) / (union + 1e-6)

        iou = calculate_iou(logits, target, valid_mask, threshold=self.threshold)

        self._log_metric(stage, name, iou, batch_size=batch_size, on_step=True, on_epoch=True, prog_bar=(stage == "val"))

    def _log_metric(self, stage: str, name: str, value, **kwargs):
        if self.iteration is not None:
            self.log(f"{stage}/{self.iteration}_{name}", value, **kwargs)

        self.log(f"{stage}/{name}", value, **kwargs)

    def _pseudo_loss_weight(self):
        weight = self.pseudo_loss_weight

        if self.pseudo_loss_rampup_epochs <= 0:
            return weight

        factor = min(self.current_epoch / self.pseudo_loss_rampup_epochs, 1.0)

        return weight * factor

    def _export_validation_predictions_this_epoch(self):
            return self.current_epoch % self.val_image_every_n_epochs == 0
