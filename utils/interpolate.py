import torch.nn.functional as F


def interpolate(x, target_size):
    """
    Exact equivalent of SupervisedModule.interpolate().
    """

    _, _, height, width = x.shape

    # x = F.interpolate(x, (height // interpolation_size[0], width // interpolation_size[1]), mode="nearest")
    x = F.interpolate(x.float(), size=target_size, mode="nearest")

    return x


def interpolate_reverse(x, target_size):
    """
    Exact equivalent of SupervisedModule.interpolate_reverse().
    """

    # x = F.interpolate(x.float(), scale_factor=interpolation_size, mode="bilinear").squeeze(0)
    x = F.interpolate(x.float(), size=target_size, mode="bilinear")

    return x