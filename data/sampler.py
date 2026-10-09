import torch
from torch.utils.data import WeightedRandomSampler

from .dataset import CombinedInkDataset


def create_sampler(
    dataset,
    mode="normal",
    num_samples=None,
    alpha=5.0,
    pos_threshold=0.05,
    real_fraction=None,
    limit_pseudo_tiles=True,
):
    """
    Creates a weighted sampler that samples approximately
    50% positive and 50% negative tiles.

    If real_fraction is given for a ConcatDataset, the sampling
    distribution is additionally balanced between real and
    pseudo-labelled samples.

    This results in:

        real_fraction / 2       real positive
        real_fraction / 2       real negative
        (1 - real_fraction) / 2 pseudo positive
        (1 - real_fraction) / 2 pseudo negative
    
    Without real_fraction:
        50% positive / 50% negative

    With real_fraction:
        real_fraction of samples are real
        (1 - real_fraction) are pseudo-labelled

        Within both groups:
            50% positive / 50% negative

        Example:
            real_fraction = 0.6

            30% real positive
            30% real negative
            20% pseudo positive
            20% pseudo negative

    Within each group, positive tiles can be weighted according
    to their positive_fraction.

    Parameters
    ----------
    dataset : InkDataset or ConcatDataset
    mode : str
        "normal", "balanced" or "sparse_positive".
    num_samples : int, optional
        Number of samples drawn by the sampler.
    alpha : float
        Controls how strongly positive_fraction influences the sampling.
        alpha=0 -> all positive tiles equally likely.
    pos_threshold : float
        Threshold above which a tile is considered positive.
    real_fraction : float, optional
        Fraction of samples that should come from real data.
        Only applies to ConcatDataset.
    limit_pseudo_tiles : bool
        If True, the number of pseudo samples is limited to the number of
        available pseudo tiles. In this case, real_fraction is treated
        as a minimum fraction of real samples.
    """

    if len(dataset) == 0:
        raise ValueError("Cannot create a sampler for an empty dataset.")

    if limit_pseudo_tiles and real_fraction is None:
        raise ValueError("limit_pseudo_tiles requires real_fraction.")

    positive = torch.tensor([tile["positive_fraction"] > pos_threshold for tile in dataset.tiles], dtype=torch.bool)
    positive_fraction = torch.tensor([tile["positive_fraction"] for tile in dataset.tiles], dtype=torch.float)

    weights = torch.ones(len(dataset), dtype=torch.float)

    match mode:
        case "balanced":
            pass
        case "normal":
            weights[positive] = 1.0 + alpha * positive_fraction[positive]
        case "sparse_positive":
            weights[positive] = 1.0 + alpha * (1.0 - positive_fraction[positive])
        case _:
            raise ValueError(f"Unknown sampling mode: {mode}")

    # ------------------------------------------------------------------
    # Determine real / pseudo-labelled samples
    # ------------------------------------------------------------------

    if real_fraction is not None:
        if not isinstance(dataset, CombinedInkDataset):
            raise ValueError("real_fraction is only supported for CombinedInkDataset.")

        if not 0 < real_fraction < 1:
            raise ValueError("real_fraction must be strictly between 0 and 1.")

        is_pseudo = torch.cat([torch.full((len(child),), child.pseudo_label_version is not None, dtype=torch.bool) for child in dataset.datasets])
        is_real = ~is_pseudo

        # --------------------------------------------------------------
        # Target probability mass for each of the four groups
        # --------------------------------------------------------------

        # Four groups:
        #
        # real positive       -> real_fraction / 2
        # real negative       -> real_fraction / 2
        # pseudo positive     -> (1-real_fraction) / 2
        # pseudo negative     -> (1-real_fraction) / 2

        if limit_pseudo_tiles:
            num_samples = num_samples if num_samples is not None else len(dataset)

            num_pseudo_positive = (is_pseudo & positive).sum().item()
            num_pseudo_negative = (is_pseudo & ~positive).sum().item()

            max_pseudo_positive_fraction = num_pseudo_positive / num_samples
            max_pseudo_negative_fraction = num_pseudo_negative / num_samples

            max_pseudo_fraction = 2 * min(max_pseudo_positive_fraction, max_pseudo_negative_fraction)

            real_fraction = max(real_fraction, 1.0 - max_pseudo_fraction)

        groups = [
            (is_real & positive, real_fraction / 2),
            (is_real & ~positive, real_fraction / 2),
            (is_pseudo & positive, (1.0 - real_fraction) / 2),
            (is_pseudo & ~positive, (1.0 - real_fraction) / 2),
        ]

        # Normalize each group independently
        for mask, target_mass in groups:
            if mask.any():
                weights[mask] *= target_mass / weights[mask].sum().clamp_min(1e-12)
            elif target_mass > 0:
                raise ValueError("Cannot satisfy real_fraction and 50/50 positive/negative sampling because one of the required groups is empty.")

    else:
        # No real/pseudo balancing requested:
        # normalize positive and negative classes to 50/50.
        if not positive.any() or positive.all():
            weights = torch.ones(len(dataset), dtype=torch.float)
        else:
            pos_sum = weights[positive].sum()
            neg_sum = weights[~positive].sum()

            weights[positive] *= 0.5 / pos_sum
            weights[~positive] *= 0.5 / neg_sum

    sampler = WeightedRandomSampler(
        weights=weights,
        num_samples=num_samples if num_samples is not None else len(dataset),
        replacement=True,
    )

    return sampler


# def create_sampler(
#     dataset,
#     mode="normal",
#     num_samples=None,
#     alpha=5.0,
#     pos_threshold=0.05,
#     # real_fraction=None,
# ):
#     """
#     Creates a weighted sampler that samples approximately
#     50% positive and 50% negative tiles.

#     Within the positive tiles, tiles with a larger positive_fraction
#     are sampled more frequently.

#     Parameters
#     ----------
#     dataset : InkDataset
#     alpha : float
#         Controls how strongly positive_fraction influences the sampling.
#         alpha=0 -> all positive tiles equally likely.
#     """

#     if len(dataset) == 0:
#         raise ValueError("Cannot create a sampler for an empty dataset.")

#     positive = torch.tensor([tile["positive_fraction"] > pos_threshold for tile in dataset.tiles], dtype=torch.bool)

#     positive_fraction = torch.tensor([tile["positive_fraction"] for tile in dataset.tiles], dtype=torch.float)

#     weights = torch.ones(len(dataset), dtype=torch.float)

#     match mode:
#         case "balanced":
#             pass
#         case "normal":
#             # Increase weights of positive samples according to their ink fraction
#             weights[positive] = 1.0 + alpha * positive_fraction[positive]
#         case "sparse_positive":
#             weights[positive] = 1.0 + alpha * (1.0 - positive_fraction[positive])

#     if not positive.any() or positive.all():
#         weights = torch.ones(len(dataset), dtype=torch.float)
#         return WeightedRandomSampler(weights, num_samples or len(dataset), replacement=True)

#     # Normalize each class separately so both classes have equal total weight
#     pos_sum = weights[positive].sum()
#     neg_sum = weights[~positive].sum()

#     weights[positive] *= 0.5 / pos_sum
#     weights[~positive] *= 0.5 / neg_sum

#     # if real_fraction is not None and isinstance(dataset, ConcatDataset):
#     #     if not 0 < real_fraction < 1:
#     #         raise ValueError("real_fraction must be strictly between 0 and 1.")
#     #     is_pseudo = torch.cat([
#     #         torch.full(
#     #             (len(child),),
#     #             bool(getattr(child, "use_pseudo_labels", False) or getattr(child, "mark_as_pseudo", False)),
#     #         )
#     #         for child in dataset.datasets
#     #     ])
#     #     # Explicitly reserve sampling mass for real and pseudo-labelled data.
#     #     # This avoids the larger source silently dominating a ConcatDataset.
#     #     if is_pseudo.any() and (~is_pseudo).any():
#     #         for group, target_mass in ((~is_pseudo, real_fraction), (is_pseudo, 1 - real_fraction)):
#     #             weights[group] *= target_mass / weights[group].sum().clamp_min(1e-12)

#     sampler = WeightedRandomSampler(
#         weights=weights,
#         num_samples=num_samples if num_samples is not None else len(dataset),
#         replacement=True,
#     )

#     return sampler
