import lightning as pl
import torch

from utils import PredictionMerger, save_predictions_as_images


class LitModel(pl.LightningModule):
    def __init__(self, model, loss, lr, warmup_epochs, max_epochs, weight_decay, val_img_dir, val_image_every_n_epochs, batch_size, iteration=None):
        super().__init__()

        self.lr = lr
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.iteration = iteration
        self.val_image_every_n_epochs = val_image_every_n_epochs

        # self.save_hyperparameters(ignore=["model"])
        self.save_hyperparameters(ignore=["model", "loss"])

        self.model = model
        self.loss = loss

        self.prediction_merger = PredictionMerger()

        self.val_img_dir = val_img_dir

    def forward(self, x):
        return self.model(x)

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        """Keep metadata on CPU and make pinned-memory transfers asynchronous."""
        for key in ("tile", "label", "mask"):
            if batch[key] is not None:
                batch[key] = batch[key].to(device, non_blocking=True)
        return batch

    def _export_validation_predictions_this_epoch(self):
        return self.current_epoch % self.val_image_every_n_epochs == 0

    def training_step(self, batch, batch_idx):
        # tile, label, mask, is_pseudo, segment, x, y, segment_shape = batch
        tile = batch["tile"]
        label = batch["label"]
        mask = batch["mask"]

        output = self(tile)
        loss = self.loss(output, label, mask)

        if self.iteration is not None:
            self.log(f"train/{str(self.iteration)}_loss", loss, batch_size=self.batch_size, on_step=True, on_epoch=True, prog_bar=True)
            self.log(f"train/loss", loss, batch_size=self.batch_size, on_step=True, on_epoch=True, prog_bar=True)
        else:
            self.log(f"train/loss", loss, batch_size=self.batch_size, on_step=True, on_epoch=True, prog_bar=True)

        self._log_tile_dice("train", output, label, mask, self.batch_size)
            
        return loss

    def validation_step(self, batch, batch_idx):
        # tile, label, mask, is_pseudo, segment, x, y, segment_shape = batch
        tile = batch["tile"]
        label = batch["label"]
        mask = batch["mask"]
        segment = batch["segment"]
        x = batch["x"]
        y = batch["y"]
        shape = batch["shape"]

        output = self(tile)
        loss = self.loss(output, label, mask)

        if self._export_validation_predictions_this_epoch():
            probs = torch.sigmoid(output).squeeze(1).detach().float().cpu().numpy()
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

        if self.iteration is not None:
            self.log(f"val/{str(self.iteration)}_loss", loss, batch_size=self.batch_size, on_step=True, on_epoch=True, prog_bar=True)
            self.log(f"val/loss", loss, batch_size=self.batch_size, on_step=True, on_epoch=True, prog_bar=True)
        else:
            self.log(f"val/loss", loss, batch_size=self.batch_size, on_step=True, on_epoch=True, prog_bar=True)

        self._log_tile_dice("val", output, label, mask, self.batch_size)

        return loss

    def _log_tile_dice(self, stage, logits, target, valid_mask, batch_size):
        prediction = (torch.sigmoid(logits) >= 0.5).to(target.dtype) * valid_mask
        target = target * valid_mask
        intersection = (prediction * target).sum()
        dice = (2 * intersection + 1e-6) / (prediction.sum() + target.sum() + 1e-6)

        self.log(f"{stage}/tile_dice", dice, batch_size=batch_size, on_step=False, on_epoch=True, prog_bar=(stage == "val"))

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
