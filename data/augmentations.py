import torch
import torch.nn.functional as F
from torchvision.transforms import v2


class AddGaussianNoise(torch.nn.Module):
    def __init__(self, mean=0.0, sigma=0.05):
        super().__init__()

        self.mean = mean
        self.sigma = sigma

    def forward(self, image, *masks):
        noise = torch.randn_like(image)
        image = image + noise * self.sigma + self.mean
        image = torch.clamp(image, 0, 1)
        return (image, *masks)


class AdditiveBrightness(torch.nn.Module):
    """Add a constant brightness offset."""

    def __init__(self, min_value=-0.15, max_value=0.15):
        super().__init__()

        self.min_value = min_value
        self.max_value = max_value

    def forward(self, image, *masks):
        value = torch.empty(
            1,
            device=image.device,
            dtype=image.dtype,
        ).uniform_(self.min_value, self.max_value)

        image = torch.clamp(image + value, 0, 1)
        return (image, *masks)


class GammaTransform(torch.nn.Module):
    """Apply a random gamma correction."""

    def __init__(self, gamma=(0.7, 1.5)):
        super().__init__()

        self.gamma = gamma

    def forward(self, image, *masks):
        gamma = torch.empty(1, device=image.device, dtype=image.dtype).uniform_(*self.gamma)

        image = torch.clamp(image, 0, 1)
        image = image.pow(gamma)

        return (image, *masks)


class MultiplicativeBrightness(torch.nn.Module):
    """Multiply image intensities by a random factor."""

    def __init__(self, multiplier=(0.75, 1.25)):
        super().__init__()

        self.multiplier = multiplier

    def forward(self, image, *masks):
        factor = torch.empty( 1, device=image.device, dtype=image.dtype).uniform_(*self.multiplier)

        image = torch.clamp(image * factor, 0, 1)

        return (image, *masks)


class BlankRectangle(torch.nn.Module):
    """Replace a random rectangle with its local mean."""

    def __init__(self, rectangle_size, num_rectangles=(1, 5)):
        super().__init__()

        self.rectangle_size = rectangle_size
        self.num_rectangles = num_rectangles

    def forward(self, image, *masks):
        image = image.clone()

        # Assumes image shape (..., H, W).
        height, width = image.shape[-2:]

        num_rectangles = torch.randint(self.num_rectangles[0], self.num_rectangles[1] + 1, (1,), device=image.device).item()

        for _ in range(num_rectangles):
            rect_h = min(torch.randint(self.rectangle_size[-2][0], self.rectangle_size[-2][1] + 1, (1,), device=image.device).item(), height)
            rect_w = min(torch.randint( self.rectangle_size[-1][0], self.rectangle_size[-1][1] + 1, (1,), device=image.device).item(), width)

            y = torch.randint(0, max(1, height - rect_h + 1), (1,), device=image.device).item()
            x = torch.randint(0, max(1, width - rect_w + 1), (1,), device=image.device).item()

            value = image[..., y : y + rect_h, x : x + rect_w].mean()

            image[..., y : y + rect_h, x : x + rect_w] = value

        return (image, *masks)


class RandomLowResolution(torch.nn.Module):
    """Simulate lower image resolution."""

    def __init__(self, scale=(0.25, 1.0)):
        super().__init__()

        self.scale = scale

    def forward(self, image, *masks):
        height, width = image.shape[-2:]

        scale = (torch.empty( 1, device=image.device, dtype=image.dtype,).uniform_(*self.scale).item())

        if scale >= 1.0:
            return (image, *masks)

        small_height = max(1, round(height * scale))
        small_width = max(1, round(width * scale))

        image = torch.nn.functional.interpolate(
            image.unsqueeze(0) if image.ndim == 3 else image,
            size=(small_height, small_width),
            mode="bilinear",
            align_corners=False,
        )

        image = torch.nn.functional.interpolate(
            image,
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )

        if image.ndim == 4:
            image = image.squeeze(0)

        return (image, *masks)


class TransposeAxes(torch.nn.Module):
    """Randomly transpose the spatial axes."""

    def __init__(self, axes=(0, 1)):
        super().__init__()

        self.axes = axes

    def forward(self, image, *masks):
        if len(self.axes) < 2:
            return (image, *masks)

        axis_a, axis_b = self.axes[:2]

        image = torch.swapaxes(image, axis_a, axis_b)
        masks = tuple(torch.swapaxes(mask, axis_a, axis_b) for mask in masks)

        return (image, *masks)


class InvertImage(torch.nn.Module):
    """Invert image intensities."""

    def forward(self, image, *masks):
        return (1.0 - image, *masks)


class MotionBlur(torch.nn.Module):
    """Apply a randomly oriented motion-blur kernel."""

    def __init__(self, kernel_size=7):
        super().__init__()

        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd")

        self.kernel_size = kernel_size

    def forward(self, image, *masks):
        kernel_size = self.kernel_size

        # Horizontal motion-blur kernel.
        kernel = torch.zeros((1, 1, kernel_size, kernel_size), device=image.device, dtype=image.dtype)
        kernel[:, :, kernel_size // 2, :] = 1.0 / kernel_size

        # Randomly rotate the kernel by 0, 90, 180 or 270 degrees.
        k = torch.randint(0, 4, ()).item()
        kernel = torch.rot90(kernel, k=k, dims=(-2, -1))

        # Convert image to NCHW.
        original_ndim = image.ndim

        if original_ndim == 2:
            image = image.unsqueeze(0).unsqueeze(0)
        elif original_ndim == 3:
            image = image.unsqueeze(0)
        elif original_ndim != 4:
            raise ValueError(f"Expected image with 2, 3 or 4 dimensions, got {original_ndim}")

        channels = image.shape[1]
        kernel = kernel.expand(channels, 1, -1, -1)

        image = F.conv2d( image, kernel, padding=kernel_size // 2, groups=channels)
        image = torch.clamp(image, 0, 1)

        if original_ndim == 2:
            image = image.squeeze(0).squeeze(0)
        elif original_ndim == 3:
            image = image.squeeze(0)

        return (image, *masks)


class CoarseDropout(torch.nn.Module):
    """
    Cutout / coarse dropout.

    Creates up to `max_holes` rectangular holes. Each dimension of a
    hole is at most `max_size_fraction` of the corresponding image dimension.
    """

    def __init__(
        self,
        max_holes=2,
        max_size_fraction=0.2,
        fill=0.0,
    ):
        super().__init__()

        self.max_holes = max_holes
        self.max_size_fraction = max_size_fraction
        self.fill = fill

    def forward(self, image, *masks):
        image = image.clone()

        height = image.shape[-2]
        width = image.shape[-1]

        max_height = max(1, int(height * self.max_size_fraction))
        max_width = max(1, int(width * self.max_size_fraction))

        num_holes = torch.randint(1, self.max_holes + 1, (), device=image.device).item()

        for _ in range(num_holes):
            hole_height = torch.randint(1, max_height + 1, (), device=image.device).item()
            hole_width = torch.randint(1, max_width + 1, (), device=image.device).item()
            top = torch.randint(0, max(1, height - hole_height + 1), (), device=image.device).item()
            left = torch.randint(0, max(1, width - hole_width + 1), (), device=image.device).item()

            image[..., top : top + hole_height, left : left + hole_width] = self.fill

        return (image, *masks)


class OneOf(torch.nn.Module):
    """Randomly select exactly one transform."""

    def __init__(self, transforms):
        super().__init__()
        self.transforms = torch.nn.ModuleList(transforms)

    def forward(self, image, *masks):
        index = torch.randint(len(self.transforms), (), device=image.device).item()

        return self.transforms[index](image, *masks)


def build_augmentation(aug):
    transforms = []

    def add_with_probability(transform, spec):
        """Apply a transform only with the probability from the YAML."""
        p = getattr(spec, "p", 1.0)

        if p < 1.0:
            transforms.append(v2.RandomApply([transform], p=p))
        else:
            transforms.append(transform)

    for t in aug.transforms:
        match t.name:
            # ---------------------------------------------------------
            # Spatial augmentations
            # ---------------------------------------------------------

            case "random_horizontal_flip":
                transforms.append(v2.RandomHorizontalFlip(p=t.p))
            case "random_vertical_flip":
                transforms.append(v2.RandomVerticalFlip(p=t.p))
            case "random_rotation":
                add_with_probability(v2.RandomRotation(degrees=t.degrees), t)
            case "random_affine":
                add_with_probability(v2.RandomAffine(degrees=t.degrees, translate=t.translate, scale=t.scale, shear=t.shear), t)
            case "random_resized_crop":
                add_with_probability(v2.RandomResizedCrop(size=t.size, scale=t.scale, ratio=t.ratio), t)
            case "transpose_axes":add_with_probability(TransposeAxes(axes=t.axes), t)

            # ---------------------------------------------------------
            # Intensity augmentations
            # ---------------------------------------------------------

            case "color_jitter":
                add_with_probability(v2.ColorJitter(brightness=t.brightness, contrast=t.contrast, saturation=t.saturation, hue=t.hue), t)
            case "contrast":
                add_with_probability(v2.RandomAutocontrast(p=1.0), t)
            case "multiplicative_brightness":
                add_with_probability(MultiplicativeBrightness(multiplier=t.multiplier), t)
            case "brightness_additive":
                add_with_probability(AdditiveBrightness(min_value=t.min_value, max_value=t.max_value), t)
            case "gamma":
                add_with_probability(GammaTransform(gamma=t.gamma), t)
            case "invert":
                add_with_probability(InvertImage(), t)
            case "sharpening":
                add_with_probability(v2.RandomAdjustSharpness(sharpness_factor=t.strength, p=1.0), t)

            # ---------------------------------------------------------
            # Noise / degradation
            # ---------------------------------------------------------

            case "gaussian_blur":
                add_with_probability(v2.GaussianBlur(kernel_size=t.kernel_size, sigma=t.sigma), t)
            case "gaussian_noise":
                add_with_probability(AddGaussianNoise(mean=t.mean, sigma=t.sigma), t)
            case "low_resolution":
                add_with_probability(RandomLowResolution(scale=t.scale), t)
            case "blank_rectangle":
                add_with_probability(BlankRectangle(rectangle_size=t.rectangle_size, num_rectangles=t.num_rectangles), t)
            case "motion_blur":
                add_with_probability(MotionBlur(kernel_size=t.kernel_size), t)

            # ---------------------------------------------------------
            # 3D-specific augmentations
            # ---------------------------------------------------------

            case "smear":
                raise NotImplementedError("The 'smear' augmentation is 3D-specific and requires a dedicated implementation.")
            case "inhomogeneous_slice_illumination":
                raise NotImplementedError("The 'inhomogeneous_slice_illumination' augmentation requires a dedicated 3D implementation.")
            case "brightness_gradient":
                raise NotImplementedError("The 'brightness_gradient' augmentation requires a dedicated implementation.")

            # ---------------------------------------------------------
            # Cutout
            # ---------------------------------------------------------

            case "cutout":
                add_with_probability(CoarseDropout(max_holes=t.max_holes, max_size_fraction=t.max_size_fraction, fill=t.fill), t)

            # ---------------------------------------------------------
            # OneOf
            # ---------------------------------------------------------

            case "oneof":
                alternatives = []
                for alternative in t.transforms:
                    alternatives.append(build_augmentation(alternative))
                add_with_probability(OneOf(alternatives), t)

            # ---------------------------------------------------------
            # _
            # ---------------------------------------------------------

            case _:
                raise ValueError(f"Unknown transform: {t.name}")

    if not transforms:
        return v2.Identity()

    return v2.Compose(transforms)
