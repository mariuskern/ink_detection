from pathlib import Path

from.utils import train_one_iteration


def train_supervised(cfg, logger):
    output_dir = Path(cfg.path.output_dir)

    train_one_iteration(cfg, logger=logger, output_dir=output_dir)
