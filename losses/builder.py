from .dice_bce import DiceBCELoss


def build_loss(cfg):
    match cfg.loss.name:
        case "dice_bce":
            return DiceBCELoss(
                w=cfg.loss.w,
                eps=cfg.loss.eps,
                bce_label_smoothing=cfg.loss.bce_label_smoothing,
                dice_label_smoothing=cfg.loss.dice_label_smoothing
            )