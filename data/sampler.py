import torch
from torch.utils.data import ConcatDataset, WeightedRandomSampler


def create_sampler(
    dataset,
    mode="normal",
    num_samples=None,
    alpha=5.0,
    pos_threshold=0.05,
    # real_fraction=None,
):
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

    if len(dataset) == 0:
        raise ValueError("Cannot create a sampler for an empty dataset.")

    positive = torch.tensor([tile["positive_fraction"] > pos_threshold for tile in dataset.tiles], dtype=torch.bool)

    positive_fraction = torch.tensor([tile["positive_fraction"] for tile in dataset.tiles], dtype=torch.float)

    weights = torch.ones(len(dataset), dtype=torch.float)

    match mode:
        case "balanced":
            pass
        case "normal":
            # Increase weights of positive samples according to their ink fraction
            weights[positive] = 1.0 + alpha * positive_fraction[positive]
        case "sparse_positive":
            weights[positive] = 1.0 + alpha * (1.0 - positive_fraction[positive])

    if not positive.any() or positive.all():
        weights = torch.ones(len(dataset), dtype=torch.float)
        return WeightedRandomSampler(weights, num_samples or len(dataset), replacement=True)

    # Normalize each class separately so both classes have equal total weight
    pos_sum = weights[positive].sum()
    neg_sum = weights[~positive].sum()

    weights[positive] *= 0.5 / pos_sum
    weights[~positive] *= 0.5 / neg_sum

    # if real_fraction is not None and isinstance(dataset, ConcatDataset):
    #     if not 0 < real_fraction < 1:
    #         raise ValueError("real_fraction must be strictly between 0 and 1.")
    #     is_pseudo = torch.cat([
    #         torch.full(
    #             (len(child),),
    #             bool(getattr(child, "use_pseudo_labels", False) or getattr(child, "mark_as_pseudo", False)),
    #         )
    #         for child in dataset.datasets
    #     ])
    #     # Explicitly reserve sampling mass for real and pseudo-labelled data.
    #     # This avoids the larger source silently dominating a ConcatDataset.
    #     if is_pseudo.any() and (~is_pseudo).any():
    #         for group, target_mass in ((~is_pseudo, real_fraction), (is_pseudo, 1 - real_fraction)):
    #             weights[group] *= target_mass / weights[group].sum().clamp_min(1e-12)

    sampler = WeightedRandomSampler(
        weights=weights,
        num_samples=num_samples if num_samples is not None else len(dataset),
        replacement=True,
    )

    return sampler
