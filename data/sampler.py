import torch
from torch.utils.data import WeightedRandomSampler


def create_balanced_sampler(dataset, alpha=5.0):
    """
    Creates a weighted sampler that samples approximately
    50% positive and 50% negative tiles.

    Within the positive tiles, tiles with a larger positive_fraction
    are sampled more frequently.

    Parameters
    ----------
    dataset : InkDataset
    alpha : float
        Controls how strongly positive_fraction influences the sampling.
        alpha=0 -> all positive tiles equally likely.
    """

    positive = torch.tensor([tile["positive_fraction"] > 0 for tile in dataset.tiles], dtype=torch.bool)

    positive_fraction = torch.tensor([tile["positive_fraction"] for tile in dataset.tiles], dtype=torch.float)

    weights = torch.ones(len(dataset), dtype=torch.float)

    # Increase weights of positive samples according to their ink fraction
    weights[positive] = 1.0 + alpha * positive_fraction[positive]

    # Normalize each class separately so both classes have equal total weight
    pos_sum = weights[positive].sum()
    neg_sum = weights[~positive].sum()

    weights[positive] *= 0.5 / pos_sum
    weights[~positive] *= 0.5 / neg_sum

    sampler = WeightedRandomSampler(
        weights=weights,
        num_samples=len(dataset),
        replacement=True,
    )

    return sampler
