import torch

from monai.data import MetaTensor
from monai.transforms import Invertd
from monai.transforms import MapTransform, InvertibleTransform
from monai.config import KeysCollection
import monai.transforms as mt


class MonaiTransformWrapper:
    """
    MONAI wrapper with a torchvision-like external interface.

    External interface
    ------------------
    image:
        (C, D, H, W)

    mask:
        (C, H, W)

    Internally MONAI receives
    -------------------------
    image:
        (C, H, W, D)

    mask:
        (C, H, W, 1)

    The transform state belonging to each call is returned explicitly
    and can later be used to invert a prediction.

    Example
    -------
        image_t, mask_t, state = transform(image, mask)

        prediction = model(image_t)

        prediction_original = transform.inverse(
            prediction,
            state,
        )
    """

    IMAGE_KEY = "image"
    MASK_KEY = "mask"

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, *masks, return_state=False):
        """
        Apply MONAI transforms.

        Parameters
        ----------
        image:
            torch.Tensor / tv_tensor.Image
            Shape: (C, D, H, W)

        masks:
            torch.Tensor / tv_tensor.Mask
            Shape: (C, H, W)

        Returns
        -------
        image_transformed:
            Shape: (C, D, H, W)

        masks_transformed:
            Shape: (C, H, W)

        state:
            Dictionary containing the transformed MetaTensors.
            This state MUST be passed to inverse().
        """

        image = torch.as_tensor(image)

        if image.ndim != 4:
            raise ValueError(f"Expected image with shape (C, D, H, W), " f"got {tuple(image.shape)}")

        data = {}

        # --------------------------------------------------------------
        # Image
        #
        # External:
        #   C D H W
        #
        # MONAI:
        #   C H W D
        # --------------------------------------------------------------

        image_monai = image.permute(0, 2, 3, 1).contiguous()

        data[self.IMAGE_KEY] = MetaTensor(image_monai)

        # --------------------------------------------------------------
        # Masks
        #
        # External:
        #   C H W
        #
        # MONAI:
        #   C H W 1
        #
        # The singleton D dimension makes the mask compatible with
        # 3D MONAI transforms.
        # --------------------------------------------------------------

        if masks:
            transformed_masks = []

            for mask in masks:
                mask = torch.as_tensor(mask)

                if mask.ndim != 3:
                    raise ValueError(f"Expected mask with shape (C, H, W), got {tuple(mask.shape)}")

                mask_monai = mask.unsqueeze(-1).contiguous()

                transformed_masks.append(MetaTensor(mask_monai))

            # Only one mask key is used internally.
            #
            # If multiple masks are passed, stack them along channel.
            #
            # Example:
            #   mask1: (C,H,W,1)
            #   mask2: (C,H,W,1)
            #
            # becomes:
            #   (N*C,H,W,1)
            #
            # For the usual case of one mask this is simply (C,H,W,1).

            if len(transformed_masks) == 1:
                data[self.MASK_KEY] = transformed_masks[0]
            else:
                data[self.MASK_KEY] = MetaTensor(torch.cat(transformed_masks, dim=0))

        # --------------------------------------------------------------
        # Apply MONAI transforms
        # --------------------------------------------------------------

        data = self.transforms(data)

        # --------------------------------------------------------------
        # Image back to external representation
        #
        # MONAI:
        #   C H W D
        #
        # External:
        #   C D H W
        # --------------------------------------------------------------

        transformed_image = data[self.IMAGE_KEY]
        transformed_image = transformed_image.permute(0, 3, 1, 2).contiguous()
        transformed_image = transformed_image.as_tensor()

        # --------------------------------------------------------------
        # Masks back to external representation
        #
        # MONAI:
        #   C H W 1
        #
        # External:
        #   C H W
        # --------------------------------------------------------------

        transformed_masks = []

        if masks:
            transformed_mask = data[self.MASK_KEY]

            # Remove singleton D dimension.
            #
            # C H W 1 -> C H W
            transformed_mask = transformed_mask[..., 0]
            transformed_mask = transformed_mask.as_tensor()

            # If multiple masks were supplied, they are currently
            # concatenated along C. Split them back.
            if len(masks) == 1:
                transformed_masks.append(transformed_mask)
            else:
                channel_sizes = [torch.as_tensor(mask).shape[0] for mask in masks]

                transformed_masks = list(torch.split(transformed_mask, channel_sizes, dim=0))

        # --------------------------------------------------------------
        # IMPORTANT:
        #
        # Keep the MetaTensor itself as the state.
        #
        # It contains MONAI's applied_operations history.
        # --------------------------------------------------------------

        state = {
            self.IMAGE_KEY: data[self.IMAGE_KEY],
        }

        if masks:
            state[self.MASK_KEY] = data[self.MASK_KEY]

        # --------------------------------------------------------------
        # Return
        # --------------------------------------------------------------

        if masks:
            if return_state:
                return transformed_image, *transformed_masks, state
            return transformed_image, *transformed_masks

        if return_state:
            return transformed_image, state

        return transformed_image

    def inverse(self, tensor, state, key="prediction", nearest_interp=True):
        tensor = torch.as_tensor(tensor)

        if tensor.ndim != 3:
            raise ValueError(f"Expected prediction with shape (C, H, W), got {tuple(tensor.shape)}")

        # 2D prediction -> MONAI-compatible pseudo-3D tensor
        prediction_monai = tensor.unsqueeze(-1).contiguous()

        data = {
            self.IMAGE_KEY: state[self.IMAGE_KEY],
            key: MetaTensor(prediction_monai),
        }

        inv = Invertd(
            keys=key,
            transform=self.transforms,
            orig_keys=self.IMAGE_KEY,
            nearest_interp=nearest_interp,
        )
        
        data = inv(data)

        prediction = data[key]

        # C,H,W,1 -> C,H,W
        prediction = prediction[..., 0].contiguous()
        prediction = prediction.as_tensor()

        return prediction

    def __repr__(self):
        return f"{self.__class__.__name__}(" f"{self.transforms!r})"


class TTANotSupported(MapTransform, InvertibleTransform):
    """
    Wrapper for transforms that are valid during forward/training,
    but must not be inverted for TTA.

    The wrapped transform is executed normally. The wrapper adds
    itself to MONAI's applied_operations history. If MONAI later
    tries to invert the transform, an explicit RuntimeError is raised.
    """

    def __init__(self, transform, keys: KeysCollection = ["image", "mask"], name: str | None = None, allow_missing_keys: bool = True):
        MapTransform.__init__(self, keys=keys, allow_missing_keys=allow_missing_keys)
        InvertibleTransform.__init__(self)

        self.transform = transform
        self.name = name or type(transform).__name__

    def __call__(self, data):
        data = self.transform(data)

        for key in self.key_iterator(data):
            self.push_transform(data[key], extra_info={"name": self.name})

        return data

    def inverse(self, data):
        raise RuntimeError(f"Transform '{self.name}' is not compatible with TTA because it cannot be spatially inverted.")

    def __repr__(self):
        return f"{self.__class__.__name__}(transform={self.transform!r}, name={self.name!r})"


class RandResizedCrop2D(MapTransform, InvertibleTransform):
    """
    RandomResizedCrop operating only on H/W.

    Input:
        (C, H, W, D)

    D is treated as an independent slice dimension and is never
    cropped or resized.

    The transform is invertible in the sense that the prediction
    can be mapped back into the original H/W canvas. Pixels outside
    the crop cannot be reconstructed and are therefore filled with 0.
    """

    def __init__(self, keys, size, scale=(0.08, 1.0), ratio=(3 / 4, 4 / 3), mode=("bilinear", "nearest"), allow_missing_keys=False):
        MapTransform.__init__(self, keys, allow_missing_keys)
        InvertibleTransform.__init__(self)

        if isinstance(size, int):
            size = (size, size)

        self.size = tuple(size)
        self.scale = scale
        self.ratio = ratio
        self.mode = mode

    def _resize_hw(self, x, size, mode):
        """
        Resize H/W while keeping D untouched.

        x:
            (C, H, W, D)

        returns:
            (C, H_new, W_new, D)
        """

        C, H, W, D = x.shape
        
        x = x.permute(0, 3, 1, 2) # C,H,W,D -> C,D,H,W
        x = x.reshape(C * D, H, W) # C,D,H,W -> (C*D),H,W
        x = x.unsqueeze(1) # Add batch dimension for interpolate

        align_corners = False if mode in {"linear", "bilinear", "bicubic", "trilinear"} else None # (C*D),1,H,W

        x = torch.nn.functional.interpolate(x, size=size, mode=mode, align_corners=align_corners)

        x = x[:, 0] # (C*D),1,H,W -> (C*D),H,W

        H_new, W_new = size
        
        x = x.reshape(C, D, H_new, W_new) # (C*D),H,W -> C,D,H,W
        x = x.permute(0, 2, 3, 1) # C,D,H,W -> C,H,W,D

        return x

    def __call__(self, data):
        d = dict(data)

        image = d[self.keys[0]]

        if image.ndim != 4:
            raise ValueError(f"Expected image (C,H,W,D), got {image.shape}")

        _, H, W, D = image.shape

        # ----------------------------------------------------------
        # Sample crop
        # ----------------------------------------------------------

        area = H * W

        log_ratio = (
            torch.log(torch.tensor(self.ratio[0])),
            torch.log(torch.tensor(self.ratio[1])),
        )

        params = None

        for _ in range(10):
            target_area = (torch.empty(1).uniform_(self.scale[0], self.scale[1]).item() * area)
            aspect_ratio = torch.exp(torch.empty(1).uniform_(log_ratio[0],log_ratio[1])).item()

            crop_h = int(round((target_area * aspect_ratio) ** 0.5))
            crop_w = int(round((target_area / aspect_ratio) ** 0.5))

            if crop_h > 0 and crop_w > 0 and crop_h <= H and crop_w <= W:
                top = torch.randint(0, H - crop_h + 1, (1,)).item()
                left = torch.randint(0, W - crop_w + 1, (1,)).item()

                params = (top, left, crop_h, crop_w)

                break

        # ----------------------------------------------------------
        # Fallback
        # ----------------------------------------------------------

        if params is None:
            in_ratio = W / H

            if in_ratio < self.ratio[0]:
                crop_w = W
                crop_h = int(round(crop_w / self.ratio[0]))
            elif in_ratio > self.ratio[1]:
                crop_h = H
                crop_w = int(round(crop_h * self.ratio[1]))
            else:
                crop_h = H
                crop_w = W

            top = (H - crop_h) // 2
            left = (W - crop_w) // 2

            params = (top, left, crop_h, crop_w)

        top, left, crop_h, crop_w = params

        # ----------------------------------------------------------
        # Transformation metadata
        # ----------------------------------------------------------

        extra_info = {
            "top": top,
            "left": left,
            "crop_h": crop_h,
            "crop_w": crop_w,
            "orig_h": H,
            "orig_w": W,
            "orig_d": D,
            "size": self.size,
        }

        # ----------------------------------------------------------
        # Apply crop + resize
        # ----------------------------------------------------------

        for i, key in enumerate(self.key_iterator(d)):
            x = d[key]

            if x.ndim != 4:
                raise ValueError(f"{key} must have shape (C,H,W,D), got {x.shape}"
                )

            x = x[:, top:top + crop_h, left:left + crop_w, :]

            mode = self.mode[i] if isinstance(self.mode, (tuple, list)) else self.mode

            x = self._resize_hw(x, self.size, mode)

            d[key] = x

            self.push_transform(d[key], extra_info=extra_info, orig_size=(H, W, D))

        return d

    def inverse(self, data):
        raise NotImplementedError(
            "random_resized_crop is not supported for TTA. "
            "RandomResizedCrop discards spatial information outside "
            "the crop and therefore cannot be fully inverted."
        )

        """
        Invert resize + crop.

        The prediction is resized back to the original crop size and
        inserted into the original H/W canvas.

        Regions outside the crop are filled with zero.
        """

        d = dict(data)

        for key in self.key_iterator(d):

            x = d[key]

            info = self.pop_transform(d[key])

            extra_info = info["extra_info"]

            top = extra_info["top"]
            left = extra_info["left"]
            crop_h = extra_info["crop_h"]
            crop_w = extra_info["crop_w"]

            orig_h = extra_info["orig_h"]
            orig_w = extra_info["orig_w"]

            # ------------------------------------------------------
            # Resize prediction from output size back to crop size.
            #
            # x:
            #   C,H,W,D
            # ------------------------------------------------------

            x = self._resize_hw(x, (crop_h, crop_w), mode="nearest")

            # ------------------------------------------------------
            # Insert into original canvas.
            # ------------------------------------------------------

            restored = torch.zeros((x.shape[0], orig_h, orig_w, x.shape[3]), dtype=x.dtype, device=x.device)
            restored[:, top:top + crop_h, left:left + crop_w, :] = x

            d[key] = restored

        return d


def build_augmentation(aug, wrap=True):
    transforms = []

    for t in aug.transforms:
        match t.name:

            # =========================================================
            # Spatial augmentations
            # =========================================================

            case "random_horizontal_flip":
                transforms.append(
                    mt.RandFlipd(
                        keys=["image", "mask"],
                        prob=t.p,
                        spatial_axis=1,
                        allow_missing_keys=True,
                    )
                )

            case "random_vertical_flip":
                transforms.append(
                    mt.RandFlipd(
                        keys=["image", "mask"],
                        prob=t.p,
                        spatial_axis=0,
                        allow_missing_keys=True,
                    )
                )

            case "random_rotation":
                degrees = t.degrees

                if isinstance(degrees, (int, float)):
                    degrees = (-degrees, degrees)

                transforms.append(
                    TTANotSupported(
                        mt.RandRotated(
                            keys=["image", "mask"],
                            prob=t.p,
                            range_x=degrees,
                            range_y=0,
                            range_z=0,
                            mode=("bilinear", "nearest"),
                            padding_mode="border",
                            allow_missing_keys=True,
                        )
                    )
                )

            case "random_affine":
                degrees = t.degrees
                if isinstance(degrees, (int, float)):
                    degrees = (-degrees, degrees)

                # transforms.append(
                #     mt.RandAffined(
                #         keys=["image", "mask"],
                #         prob=t.p,
                #         rotate_range=degrees,
                #         translate_range=t.translate,
                #         scale_range=t.scale,
                #         shear_range=t.shear,
                #         mode=("bilinear", "nearest"),
                #         padding_mode="border",
                #         allow_missing_keys=True
                #     )
                # )

                transforms.append(
                    TTANotSupported(
                        mt.RandAffined(
                            keys=["image", "mask"],
                            prob=t.p,
                            rotate_range=(0, 0, *degrees),
                            translate_range=(
                                t.translate[0],
                                t.translate[1],
                                0,
                            ),
                            scale_range=(
                                t.scale[0],
                                t.scale[1],
                                0,
                            ),
                            shear_range=(0, 0, 0, 0, 0, 0),
                            mode=("bilinear", "nearest"),
                            padding_mode="border",
                            allow_missing_keys=True,
                        )
                    )
                )

            case "random_resized_crop":
                # # MONAI does not have a direct 3D equivalent of
                # # torchvision's RandomResizedCrop.
                # #
                # # RandZoomd + RandSpatialCropd is the closest
                # # 3D-oriented alternative.

                # transforms.append(
                #     mt.RandZoomd(
                #         keys=["image", "mask"],
                #         prob=t.p,
                #         min_zoom=min(t.scale),
                #         max_zoom=max(t.scale),
                #         mode=("trilinear", "nearest"),
                #         padding_mode="edge",
                #         allow_missing_keys=True
                #     )
                # )

                # transforms.append(
                #     mt.RandSpatialCropd(
                #         keys=["image", "mask"],
                #         roi_size=t.size,
                #         random_center=True,
                #         random_size=False,
                #         allow_missing_keys=True
                #     )
                # )

                transforms.append(
                    TTANotSupported(
                        RandResizedCrop2D(
                            keys=["image", "mask"],
                            size=t.size,
                            scale=t.scale,
                            ratio=t.ratio,
                            mode=("bilinear", "nearest"),
                            allow_missing_keys=True,
                        )
                    )
                )

            # case "transpose_axes":
            #     transforms.append(
            #         # mt.Transposed(
            #         #     keys=["image", "mask"],
            #         #     indices=t.axes,
            #         #     allow_missing_keys=True
            #         # )
            #         mt.Compose([
            #             mt.Flip(
            #                 keys=["image", "mask"],
            #                 spatial_axis=1,
            #             ),
            #             mt.Rotate90(
            #                 keys=["image", "mask"],
            #                 k=1,
            #                 spatial_axes=(0, 1),
            #             ),
            #         ])
            #     )

            case "random_axis_flip":
                transforms.append(
                    TTANotSupported(
                        mt.RandAxisFlipd(
                            keys=["image", "mask"],
                            prob=t.p,
                            spatial_axis=t.axes,
                            allow_missing_keys=True,
                        )
                    )
                )

            # =========================================================
            # Intensity augmentations
            # =========================================================

            case "contrast":
                transforms.append(
                    TTANotSupported(
                        mt.RandAdjustContrastd(
                            keys=["image"],
                            prob=t.p,
                            gamma=(0.8, 1.2),
                            allow_missing_keys=True,
                        )
                    )
                )

            case "multiplicative_brightness":
                transforms.append(
                    TTANotSupported(
                        mt.RandScaleIntensityd(
                            keys=["image"],
                            prob=t.p,
                            factors=t.multiplier,
                            allow_missing_keys=True,
                        )
                    )
                )

            case "brightness_additive":
                transforms.append(
                    TTANotSupported(
                        mt.RandShiftIntensityd(
                            keys=["image"],
                            prob=t.p,
                            offsets=(t.min_value, t.max_value),
                            allow_missing_keys=True,
                        )
                    )
                )

            case "gamma":
                transforms.append(
                    TTANotSupported(
                        mt.RandGammaCorrectiond(
                            keys=["image"],
                            prob=t.p,
                            gamma=t.gamma,
                            allow_missing_keys=True
                        )
                    )
                )

            case "invert":
                transforms.append(
                    TTANotSupported(
                        mt.RandScaleIntensityd(
                            keys=["image"],
                            prob=t.p,
                            factors=-1.0,
                            allow_missing_keys=True
                        )
                    )
                )

            case "sharpening":
                transforms.append(
                    TTANotSupported(
                        mt.RandGaussianSharpend(
                            keys=["image"],
                            prob=t.p,
                            sigma1=0.5,
                            sigma2=1.0,
                            alpha=(0.5, t.strength),
                            allow_missing_keys=True,
                        )
                    )
                )

            # =========================================================
            # Noise / degradation
            # =========================================================

            case "gaussian_blur":
                transforms.append(
                    TTANotSupported(
                        mt.RandGaussianSmoothd(
                            keys=["image"],
                            prob=t.p,
                            sigma_x=t.sigma,
                            sigma_y=t.sigma,
                            sigma_z=t.sigma,
                            allow_missing_keys=True,
                        )
                    )
                )

            case "gaussian_noise":
                transforms.append(
                    TTANotSupported(
                        mt.RandGaussianNoised(
                            keys=["image"],
                            prob=t.p,
                            mean=t.mean,
                            std=t.sigma,
                            allow_missing_keys=True,
                        )
                    )
                )

            case "low_resolution":
                transforms.extend(
                    [
                        TTANotSupported(
                            mt.RandZoomd(
                                keys=["image"],
                                prob=t.p,
                                min_zoom=t.scale,
                                max_zoom=t.scale,
                                mode=("trilinear", "nearest"),
                                padding_mode="edge",
                                allow_missing_keys=True,
                            )
                        ),
                        TTANotSupported(
                            mt.Zoomd(
                                keys=["image"],
                                zoom=1.0 / t.scale,
                                mode=("trilinear", "nearest"),
                                allow_missing_keys=True,
                            )
                        )
                    ]
                )

            case "motion_blur":
                transforms.append(
                    TTANotSupported(
                        mt.RandGaussianSmoothd(
                            keys=["image"],
                            prob=t.p,
                            sigma_x=t.kernel_size / 3.0,
                            sigma_y=t.kernel_size / 3.0,
                            sigma_z=t.kernel_size / 3.0,
                            allow_missing_keys=True,
                        )
                    )
                )

            # =========================================================
            # 3D CT-specific augmentations
            # =========================================================

            case "smear":
                transforms.append(
                    TTANotSupported(
                        mt.RandGaussianSmoothd(
                            keys=["image"],
                            prob=t.p,
                            sigma_x=t.sigma,
                            sigma_y=t.sigma,
                            sigma_z=t.sigma * 2,
                            allow_missing_keys=True,
                        )
                    )
                )

            case "inhomogeneous_slice_illumination":
                transforms.append(
                    TTANotSupported(
                        mt.RandBiasFieldd(
                            keys=["image"],
                            prob=t.p,
                            coeff_range=t.coeff_range,
                            allow_missing_keys=True,
                        )
                    )
                )

            case "brightness_gradient":
                transforms.append(
                    TTANotSupported(
                        mt.RandBiasFieldd(
                            keys=["image"],
                            prob=t.p,
                            coeff_range=t.coeff_range,
                            allow_missing_keys=True,
                        )
                    )
                )

            # =========================================================
            # Cutout
            # =========================================================

            case "cutout":
                transforms.append(
                    TTANotSupported(
                        mt.RandCoarseDropoutd(
                            keys=["image"],
                            prob=t.p,
                            holes=t.max_holes,
                            spatial_size=t.max_size_fraction,
                            fill_value=t.fill,
                            allow_missing_keys=True,
                        )
                    )
                )

            # =========================================================
            # OneOf
            # =========================================================

            case "oneof":
                alternatives = [
                    build_augmentation(alternative, wrap=False) for alternative in t.transforms
                ]

                transforms.append(
                    mt.OneOf(alternatives, weights=None, allow_missing_keys=True)
                )

            # =========================================================
            # Not recommended for CT
            # =========================================================

            case "color_jitter":
                raise ValueError("'color_jitter' is not suitable for CT volumes. Use contrast, brightness, or gamma instead.")

            case _:
                raise ValueError(f"Unknown transform: {t.name}")
    
    if not transforms:
        return None

    compose = mt.Compose(transforms)

    if wrap:
        return MonaiTransformWrapper(compose)

    return compose




# import torch
# import torch.nn.functional as F
# from dataclasses import dataclass
# from torchvision.transforms import v2


# class AddGaussianNoise(torch.nn.Module):
#     def __init__(self, mean=0.0, sigma=0.05):
#         super().__init__()

#         self.mean = mean
#         self.sigma = sigma

#     def forward(self, image, *masks):
#         noise = torch.randn_like(image)
#         image = image + noise * self.sigma + self.mean
#         image = torch.clamp(image, 0, 1)

#         if masks:
#             return (image, *masks)

#         return image


# class AdditiveBrightness(torch.nn.Module):
#     """Add a constant brightness offset."""

#     def __init__(self, min_value=-0.15, max_value=0.15):
#         super().__init__()

#         self.min_value = min_value
#         self.max_value = max_value

#     def forward(self, image, *masks):
#         value = torch.empty(
#             1,
#             device=image.device,
#             dtype=image.dtype,
#         ).uniform_(self.min_value, self.max_value)

#         image = torch.clamp(image + value, 0, 1)

#         if masks:
#             return (image, *masks)

#         return image


# class GammaTransform(torch.nn.Module):
#     """Apply a random gamma correction."""

#     def __init__(self, gamma=(0.7, 1.5)):
#         super().__init__()

#         self.gamma = gamma

#     def forward(self, image, *masks):
#         gamma = torch.empty(1, device=image.device, dtype=image.dtype).uniform_(*self.gamma)

#         image = torch.clamp(image, 0, 1)
#         image = image.pow(gamma)

#         if masks:
#             return (image, *masks)

#         return image


# class MultiplicativeBrightness(torch.nn.Module):
#     """Multiply image intensities by a random factor."""

#     def __init__(self, multiplier=(0.75, 1.25)):
#         super().__init__()

#         self.multiplier = multiplier

#     def forward(self, image, *masks):
#         factor = torch.empty(1, device=image.device, dtype=image.dtype).uniform_(*self.multiplier)

#         image = torch.clamp(image * factor, 0, 1)

#         if masks:
#             return (image, *masks)

#         return image


# class BlankRectangle(torch.nn.Module):
#     """Replace random rectangles with their local mean."""

#     def __init__(self, rectangle_size, num_rectangles=(1, 5)):
#         super().__init__()

#         self.rectangle_size = rectangle_size
#         self.num_rectangles = num_rectangles

#     def forward(self, image, *masks):
#         image = image.clone()

#         height, width = image.shape[-2:]

#         num_rectangles = torch.randint(self.num_rectangles[0], self.num_rectangles[1] + 1, (), device=image.device).item()

#         for _ in range(num_rectangles):
#             rect_h = min(torch.randint(self.rectangle_size[-2][0], self.rectangle_size[-2][1] + 1, (), device=image.device).item(), height)
#             rect_w = min(torch.randint( self.rectangle_size[-1][0], self.rectangle_size[-1][1] + 1, (), device=image.device).item(), width)

#             y = torch.randint(0, max(1, height - rect_h + 1), (), device=image.device).item()
#             x = torch.randint(0, max(1, width - rect_w + 1), (), device=image.device).item()

#             region = image[..., y:y + rect_h, x:x + rect_w]
#             value = region.mean(dim=(-2, -1), keepdim=True)

#             image[..., y:y + rect_h, x:x + rect_w] = value

#         if masks:
#             return (image, *masks)

#         return image


# class RandomLowResolution(torch.nn.Module):
#     """Simulate lower image resolution."""

#     def __init__(self, scale=(0.25, 1.0)):
#         super().__init__()

#         self.scale = scale

#     def forward(self, image, *masks):
#         height, width = image.shape[-2:]

#         scale = (torch.empty( 1, device=image.device, dtype=image.dtype,).uniform_(*self.scale).item())

#         if scale >= 1.0:
#             return (image, *masks)

#         small_height = max(1, round(height * scale))
#         small_width = max(1, round(width * scale))

#         original_ndim = image.ndim

#         if original_ndim == 2:
#             image = image.unsqueeze(0).unsqueeze(0)
#         elif original_ndim == 3:
#             image = image.unsqueeze(0)
#         elif original_ndim != 4:
#             raise ValueError(f"Expected image with 2, 3 or 4 dimensions, got {original_ndim}")

#         image = F.interpolate(
#             image,
#             size=(small_height, small_width),
#             mode="bilinear",
#             align_corners=False,
#         )

#         image = F.interpolate(
#             image,
#             size=(height, width),
#             mode="bilinear",
#             align_corners=False,
#         )

#         if original_ndim == 2:
#             image = image.squeeze(0).squeeze(0)
#         elif original_ndim == 3:
#             image = image.squeeze(0)

#         if masks:
#             return (image, *masks)

#         return image


# class TransposeAxes(torch.nn.Module):
#     """Randomly transpose the spatial axes."""

#     def __init__(self, axes=(-2, -1)):
#         super().__init__()

#         if len(axes) < 2:
#             raise ValueError("axes must contain at least two axes")

#         self.axes = axes

#     def forward(self, image, *masks):
#         axis_a, axis_b = self.axes[:2]

#         image = torch.swapaxes(image, axis_a, axis_b)
#         masks = tuple(torch.swapaxes(mask, axis_a, axis_b) for mask in masks)

#         if masks:
#             return (image, *masks)

#         return image


# class InvertImage(torch.nn.Module):
#     """Invert image intensities."""

#     def forward(self, image, *masks):
#         if masks:
#             return (1.0 - image, *masks)

#         return 1.0 - image


# class MotionBlur(torch.nn.Module):
#     """Apply a randomly oriented horizontal or vertical motion blur."""

#     def __init__(self, kernel_size=7):
#         super().__init__()

#         if kernel_size % 2 == 0:
#             raise ValueError("kernel_size must be odd")

#         self.kernel_size = kernel_size

#     def forward(self, image, *masks):
#         kernel_size = self.kernel_size

#         kernel = torch.zeros((1, 1, kernel_size, kernel_size), device=image.device, dtype=image.dtype)
#         kernel[:, :, kernel_size // 2, :] = 1.0 / kernel_size

#         if torch.randint(0, 2, (), device=image.device).item():
#             kernel = torch.rot90(kernel, k=1, dims=(-2, -1))

#         original_ndim = image.ndim

#         if original_ndim == 2:
#             image = image.unsqueeze(0).unsqueeze(0)
#         elif original_ndim == 3:
#             image = image.unsqueeze(0)
#         elif original_ndim != 4:
#             raise ValueError(f"Expected image with 2, 3 or 4 dimensions, got {original_ndim}")

#         channels = image.shape[1]
#         kernel = kernel.expand(channels, 1, -1, -1)

#         image = F.conv2d( image, kernel, padding=kernel_size // 2, groups=channels)
#         image = torch.clamp(image, 0, 1)

#         if original_ndim == 2:
#             image = image.squeeze(0).squeeze(0)
#         elif original_ndim == 3:
#             image = image.squeeze(0)

#         if masks:
#             return (image, *masks)

#         return image


# class CoarseDropout(torch.nn.Module):
#     """
#     Cutout / coarse dropout.

#     Creates up to `max_holes` rectangular holes. Each dimension of a
#     hole is at most `max_size_fraction` of the corresponding image dimension.
#     """

#     def __init__(
#         self,
#         max_holes=2,
#         max_size_fraction=0.2,
#         fill=0.0,
#     ):
#         super().__init__()

#         if max_holes < 1:
#             raise ValueError("max_holes must be >= 1")

#         if not 0.0 < max_size_fraction <= 1.0:
#             raise ValueError("max_size_fraction must be in (0, 1]")

#         self.max_holes = max_holes
#         self.max_size_fraction = max_size_fraction
#         self.fill = fill

#     def forward(self, image, *masks):
#         image = image.clone()

#         height = image.shape[-2]
#         width = image.shape[-1]

#         max_height = max(1, int(height * self.max_size_fraction))
#         max_width = max(1, int(width * self.max_size_fraction))

#         num_holes = torch.randint(1, self.max_holes + 1, (), device=image.device).item()

#         for _ in range(num_holes):
#             hole_height = torch.randint(1, max_height + 1, (), device=image.device).item()
#             hole_width = torch.randint(1, max_width + 1, (), device=image.device).item()
#             top = torch.randint(0, max(1, height - hole_height + 1), (), device=image.device).item()
#             left = torch.randint(0, max(1, width - hole_width + 1), (), device=image.device).item()

#             image[..., top : top + hole_height, left : left + hole_width] = self.fill

#         if masks:
#             return (image, *masks)

#         return image


# class OneOf(torch.nn.Module):
#     """Randomly select exactly one transform."""

#     def __init__(self, transforms):
#         super().__init__()

#         if not transforms:
#             raise ValueError("OneOf requires at least one transform")

#         self.transforms = torch.nn.ModuleList(transforms)

#     def forward(self, image, *masks):
#         index = torch.randint(len(self.transforms), (), device=image.device).item()

#         return self.transforms[index](image, *masks)


# # -------------------------------------------------------------------------
# # TTA
# # -------------------------------------------------------------------------


# @dataclass
# class TransformRecord:
#     transform: torch.nn.Module
#     params: dict | None = None


# class TTAPipeline(torch.nn.Module):
#     """Augmentation pipeline with optional TTA replay and inverse support."""

#     def __init__(self, transforms):
#         super().__init__()
#         self.transforms = torch.nn.ModuleList(transforms)

#     def forward(self, image, *masks):
#         for transform in self.transforms:
#             image, *masks = transform(image, *masks)

#         return (image, *masks)

#     def tta_forward(self, image):
#         records = []

#         for transform in self.transforms:
#             if hasattr(transform, "tta_forward"):
#                 image, params = transform.tta_forward(image)
#             else:
#                 image = transform(image)
#                 params = None

#             records.append(TransformRecord(transform=transform, params=params))

#         return image, records

#     def apply_tta(self, image):
#         return self.tta_forward(image)

#     def inverse(self, prediction, records):
#         for record in reversed(records):
#             if hasattr(record.transform, "inverse"):
#                 prediction = record.transform.inverse(prediction, record.params)

#         return prediction


# class TTAIdentity(torch.nn.Module):
#     """Identity transform for test-time augmentation."""

#     def __init__(self):
#         super().__init__()

#     def tta_forward(self, image):
#         return image, {}

#     def forward(self, image, *masks):
#         if masks:
#             return image, *masks

#         return image

#     def inverse(self, prediction, params):
#         return prediction


# class TTARandomHorizontalFlip(torch.nn.Module):
#     """Random horizontal flip with inverse support."""

#     def __init__(self, p=0.5):
#         super().__init__()
#         self.p = p

#     def tta_forward(self, image):
#         apply = torch.rand((), device=image.device).item() < self.p

#         if apply:
#             image = torch.flip(image, dims=(-1,))

#         return image, {"apply": apply}

#     def forward(self, image, *masks):
#         image, params = self.tta_forward(image)

#         if params["apply"]:
#             masks = tuple(torch.flip(mask, dims=(-1,)) for mask in masks)

#         if masks:
#             return (image, *masks)

#         return image

#     def inverse(self, prediction, params):
#         if params["apply"]:
#             prediction = torch.flip(prediction, dims=(-1,))

#         return prediction


# class TTARandomVerticalFlip(torch.nn.Module):
#     """Random vertical flip with inverse support."""

#     def __init__(self, p=0.5):
#         super().__init__()
#         self.p = p

#     def tta_forward(self, image):
#         apply = torch.rand((), device=image.device).item() < self.p

#         if apply:
#             image = torch.flip(image, dims=(-2,))

#         return image, {"apply": apply}

#     def forward(self, image, *masks):
#         image, params = self.tta_forward(image)

#         if params["apply"]:
#             masks = tuple(torch.flip(mask, dims=(-2,)) for mask in masks)

#         if masks:
#             return (image, *masks)

#         return image

#     def inverse(self, prediction, params):
#         if params["apply"]:
#             prediction = torch.flip(prediction, dims=(-2,))

#         return prediction


# class TTATransposeAxes(torch.nn.Module):
#     """Transpose spatial axes with inverse support."""

#     def __init__(self, axes=(0, 1)):
#         super().__init__()

#         if len(axes) < 2:
#             raise ValueError("axes must contain at least two axes")

#         self.axes = axes

#     def tta_forward(self, image):
#         axis_a, axis_b = self.axes[:2]

#         image = torch.swapaxes(image, axis_a, axis_b)

#         return image, {
#             "axis_a": axis_a,
#             "axis_b": axis_b,
#         }

#     def forward(self, image, *masks):
#         image, params = self.tta_forward(image)

#         masks = tuple(torch.swapaxes(mask, params["axis_a"], params["axis_b"]) for mask in masks)

#         if masks:
#             return (image, *masks)

#         return image

#     def inverse(self, prediction, params):
#         return torch.swapaxes(prediction, params["axis_a"], params["axis_b"])


# class TTARandomRotation(torch.nn.Module):
#     """Random rotation with inverse support."""

#     def __init__(self, degrees):
#         super().__init__()

#         self.transform = v2.RandomRotation(degrees)

#     def tta_forward(self, image):
#         raise RuntimeError("TTA is not supported for this transform. Only exactly invertible geometric transformations (horizontal flip, vertical flip, transpose) should be used for TTA.")

#     def forward(self, image, *masks):
#         return self.transform(image, *masks)

#     def inverse(self, prediction, params):
#         raise RuntimeError("TTA is not supported for this transform. Only exactly invertible geometric transformations (horizontal flip, vertical flip, transpose) should be used for TTA.")


# # class TTARandomRotation(torch.nn.Module):
# #     """Random rotation with inverse support."""

# #     def __init__(self, degrees):
# #         super().__init__()

# #         if isinstance(degrees, (int, float)):
# #             degrees = (-degrees, degrees)

# #         self.degrees = degrees

# #     def tta_forward(self, image):
# #         # raise RuntimeError("TTA is not supported for this transform. Only exactly invertible geometric transformations (horizontal flip, vertical flip, transpose) should be used for TTA.")

# #         angle = torch.empty(1, device=image.device, dtype=image.dtype).uniform_(*self.degrees).item()

# #         image = v2.functional.rotate(image, angle, interpolation=InterpolationMode.BILINEAR)

# #         return image, {
# #             "angle": angle,
# #         }

# #     def forward(self, image, *masks):
# #         image, params = self.tta_forward(image)

# #         transformed_masks = []

# #         for mask in masks:
# #             original_ndim = mask.ndim

# #             if original_ndim == 2:
# #                 mask = mask.unsqueeze(0)
# #             elif original_ndim not in (3, 4):
# #                 raise ValueError(f"Expected mask with 2, 3 or 4 dimensions, got {original_ndim}")

# #             mask = v2.functional.rotate(mask, params["angle"], interpolation=InterpolationMode.NEAREST)

# #             if original_ndim == 2:
# #                 mask = mask.squeeze(0)

# #             transformed_masks.append(mask)

# #         return (image, *transformed_masks)

# #     def inverse(self, prediction, params):
# #         raise RuntimeError("TTA is not supported for this transform. Only exactly invertible geometric transformations (horizontal flip, vertical flip, transpose) should be used for TTA.")

# #         original_ndim = prediction.ndim

# #         if original_ndim == 2:
# #             prediction = prediction.unsqueeze(0)
# #         elif original_ndim not in (3, 4):
# #             raise ValueError(f"Expected prediction with 2, 3 or 4 dimensions, got {original_ndim}")

# #         prediction = v2.functional.rotate(prediction, -params["angle"], interpolation=InterpolationMode.BILINEAR)

# #         if original_ndim == 2:
# #             prediction = prediction.squeeze(0)

# #         return prediction


# class TTARandomAffine(torch.nn.Module):
#     """
#     Random affine transform with replay support.

#     Note:
#         The inverse is an approximate inverse because affine transforms
#         involve interpolation. For exact TTA, prefer flips and transposes.
#     """

#     def __init__(self, degrees, translate=None, scale=None, shear=None):
#         super().__init__()

#         self.transform = v2.RandomAffine(
#             degrees=degrees,
#             translate=translate,
#             scale=scale,
#             shear=shear
#         )

#     def tta_forward(self, image):
#         raise RuntimeError("TTA is not supported for this transform. Only exactly invertible geometric transformations (horizontal flip, vertical flip, transpose) should be used for TTA.")

#     def forward(self, image, *masks):
#         return self.transform(image, *masks)

#     def inverse(self, prediction, params):
#         raise RuntimeError("TTA is not supported for this transform. Only exactly invertible geometric transformations (horizontal flip, vertical flip, transpose) should be used for TTA.")


# # class TTARandomAffine(torch.nn.Module):
# #     """
# #     Random affine transform with replay support.

# #     Note:
# #         The inverse is an approximate inverse because affine transforms
# #         involve interpolation. For exact TTA, prefer flips and transposes.
# #     """

# #     def __init__(self, degrees, translate=None, scale=None, shear=None):
# #         super().__init__()

# #         self.degrees = degrees
# #         self.translate = translate
# #         self.scale = scale
# #         self.shear = shear

# #     def _sample_parameters(self, image):
# #         height, width = image.shape[-2:]

# #         if isinstance(self.degrees, (int, float)):
# #             degrees = (-self.degrees, self.degrees)
# #         else:
# #             degrees = self.degrees

# #         angle = torch.empty(1, device=image.device, dtype=image.dtype).uniform_(*degrees).item()

# #         if self.translate is None:
# #             translate = [0, 0]
# #         else:
# #             translate = [
# #                 torch.empty(1, device=image.device, dtype=image.dtype).uniform_(-self.translate[0] * width, self.translate[0] * width).item(),
# #                 torch.empty(1, device=image.device, dtype=image.dtype).uniform_(-self.translate[1] * height, self.translate[1] * height).item(),
# #             ]

# #         if self.scale is None:
# #             scale = 1.0
# #         else:
# #             scale = torch.empty(1, device=image.device, dtype=image.dtype).uniform_(*self.scale).item()

# #         if self.shear is None:
# #             shear = [0.0, 0.0]
# #         elif isinstance(self.shear, (int, float)):
# #             shear = [
# #                 torch.empty(1, device=image.device, dtype=image.dtype).uniform_(-self.shear, self.shear).item(),
# #                 0.0,
# #             ]
# #         elif len(self.shear) == 2:
# #             shear = [
# #                 torch.empty(1, device=image.device, dtype=image.dtype).uniform_(*self.shear).item(),
# #                 0.0,
# #             ]
# #         else:
# #             shear = [
# #                 torch.empty(1, device=image.device, dtype=image.dtype).uniform_(*self.shear[:2]).item(),
# #                 torch.empty(1, device=image.device, dtype=image.dtype).uniform_(*self.shear[2:]).item(),
# #             ]

# #         translate = [
# #             round(translate[0]),
# #             round(translate[1]),
# #         ]

# #         return {
# #             "angle": angle,
# #             "translate": translate,
# #             "scale": scale,
# #             "shear": shear,
# #             "size": (height, width),
# #         }

# #     def tta_forward(self, image):
# #         # raise RuntimeError("TTA is not supported for this transform. Only exactly invertible geometric transformations (horizontal flip, vertical flip, transpose) should be used for TTA.")

# #         params = self._sample_parameters(image)

# #         image = v2.functional.affine(
# #             image,
# #             angle=params["angle"],
# #             translate=params["translate"],
# #             scale=params["scale"],
# #             shear=params["shear"],
# #             interpolation=InterpolationMode.BILINEAR,
# #         )

# #         return image, params

# #     def forward(self, image, *masks):
# #         image, params = self.tta_forward(image)

# #         transformed_masks = []

# #         for mask in masks:
# #             original_ndim = mask.ndim

# #             if original_ndim == 2:
# #                 mask = mask.unsqueeze(0)
# #             elif original_ndim not in (3, 4):
# #                 raise ValueError(f"Expected mask with 2, 3 or 4 dimensions, got {original_ndim}")

# #             mask = v2.functional.affine(
# #                 mask,
# #                 angle=params["angle"],
# #                 translate=params["translate"],
# #                 scale=params["scale"],
# #                 shear=params["shear"],
# #                 interpolation=InterpolationMode.NEAREST,
# #             )

# #             if original_ndim == 2:
# #                 mask = mask.squeeze(0)

# #             transformed_masks.append(mask)

# #         return (image, *transformed_masks)

# #     def inverse(self, prediction, params):
# #         raise RuntimeError("TTA is not supported for this transform. Only exactly invertible geometric transformations (horizontal flip, vertical flip, transpose) should be used for TTA.")

# #         original_ndim = prediction.ndim

# #         if original_ndim == 2:
# #             prediction = prediction.unsqueeze(0)
# #         elif original_ndim not in (3, 4):
# #             raise ValueError(f"Expected prediction with 2, 3 or 4 dimensions, got {original_ndim}")

# #         prediction = v2.functional.affine(
# #             prediction,
# #             angle=-params["angle"],
# #             translate=[
# #                 -params["translate"][0],
# #                 -params["translate"][1],
# #             ],
# #             scale=1.0 / params["scale"],
# #             shear=[
# #                 -params["shear"][0],
# #                 -params["shear"][1],
# #             ],
# #             interpolation=InterpolationMode.BILINEAR,
# #         )

# #         if original_ndim == 2:
# #             prediction = prediction.squeeze(0)

# #         return prediction


# class TTARandomResizedCrop(torch.nn.Module):
#     """Random resized crop with inverse support."""

#     def __init__(self, size, scale=(0.08, 1.0), ratio=(3.0 / 4.0, 4.0 / 3.0)):
#         super().__init__()

#         self.transform = v2.RandomResizedCrop(
#             size=size,
#             scale=scale,
#             ratio=ratio
#         )

#     def tta_forward(self, image):
#         raise RuntimeError("TTA is not supported for this transform. Only exactly invertible geometric transformations (horizontal flip, vertical flip, transpose) should be used for TTA.")

#     def forward(self, image, *masks):
#         return self.transform(image, *masks)

#     def inverse(self, prediction, params):
#         raise RuntimeError("TTA is not supported for this transform. Only exactly invertible geometric transformations (horizontal flip, vertical flip, transpose) should be used for TTA.")


# # class TTARandomResizedCrop(torch.nn.Module):
# #     """Random resized crop with inverse support."""

# #     def __init__(self, size, scale=(0.08, 1.0), ratio=(3.0 / 4.0, 4.0 / 3.0)):
# #         super().__init__()

# #         self.size = size
# #         self.scale = scale
# #         self.ratio = ratio

# #     def _sample_parameters(self, image):
# #         height, width = image.shape[-2:]
# #         area = height * width

# #         target_area = torch.empty(1, device=image.device, dtype=image.dtype).uniform_(*self.scale).item() * area

# #         log_ratio = torch.empty(
# #             1,
# #             device=image.device,
# #             dtype=image.dtype,
# #         ).uniform_(
# #             torch.log(torch.tensor(self.ratio[0], device=image.device, dtype=image.dtype)),
# #             torch.log(torch.tensor(self.ratio[1], device=image.device, dtype=image.dtype)),
# #         ).item()

# #         ratio = torch.exp(torch.tensor(log_ratio, device=image.device, dtype=image.dtype)).item()

# #         crop_height = min(int(round((target_area / ratio) ** 0.5)), height)
# #         crop_width = min(int(round((target_area * ratio) ** 0.5)), width)
# #         crop_height = max(1, crop_height)
# #         crop_width = max(1, crop_width)

# #         top = torch.randint(0, max(1, height - crop_height + 1), (), device=image.device).item()
# #         left = torch.randint(0, max(1, width - crop_width + 1), (), device=image.device).item()

# #         return {
# #             "top": top,
# #             "left": left,
# #             "height": crop_height,
# #             "width": crop_width,
# #             "original_size": (height, width),
# #         }

# #     def tta_forward(self, image):
# #         # raise RuntimeError("TTA is not supported for this transform. Only exactly invertible geometric transformations (horizontal flip, vertical flip, transpose) should be used for TTA.")

# #         params = self._sample_parameters(image)

# #         image = image[..., params["top"]:params["top"] + params["height"], params["left"]:params["left"] + params["width"]]

# #         original_ndim = image.ndim

# #         if original_ndim == 2:
# #             image = image.unsqueeze(0).unsqueeze(0)
# #         elif original_ndim == 3:
# #             image = image.unsqueeze(0)
# #         elif original_ndim != 4:
# #             raise ValueError(f"Expected image with 2, 3 or 4 dimensions, got {original_ndim}")

# #         image = F.interpolate(
# #             image,
# #             size=self.size,
# #             mode="bilinear",
# #             align_corners=False,
# #         )

# #         if original_ndim == 2:
# #             image = image.squeeze(0).squeeze(0)
# #         elif original_ndim == 3:
# #             image = image.squeeze(0)

# #         return image, params

# #     def forward(self, image, *masks):
# #         image, params = self.tta_forward(image)

# #         transformed_masks = []

# #         for mask in masks:
# #             original_ndim = mask.ndim

# #             mask = mask[..., params["top"]:params["top"] + params["height"], params["left"]:params["left"] + params["width"]]

# #             if original_ndim == 2:
# #                 mask = mask.unsqueeze(0).unsqueeze(0)
# #             elif original_ndim == 3:
# #                 mask = mask.unsqueeze(0)
# #             elif original_ndim != 4:
# #                 raise ValueError(f"Expected mask with 2, 3 or 4 dimensions, got {original_ndim}")

# #             mask = F.interpolate(
# #                 mask,
# #                 size=self.size,
# #                 mode="nearest",
# #             )

# #             if original_ndim == 2:
# #                 mask = mask.squeeze(0).squeeze(0)
# #             elif original_ndim == 3:
# #                 mask = mask.squeeze(0)

# #             transformed_masks.append(mask)

# #         return (image, *transformed_masks)

# #     def inverse(self, prediction, params):
# #         raise RuntimeError("TTA is not supported for this transform. Only exactly invertible geometric transformations (horizontal flip, vertical flip, transpose) should be used for TTA.")

# #         original_height, original_width = params["original_size"]

# #         original_ndim = prediction.ndim

# #         if original_ndim == 2:
# #             prediction = prediction.unsqueeze(0).unsqueeze(0)
# #         elif original_ndim == 3:
# #             prediction = prediction.unsqueeze(0)
# #         elif original_ndim != 4:
# #             raise ValueError(f"Expected prediction with 2, 3 or 4 dimensions, got {original_ndim}")

# #         prediction = F.interpolate(
# #             prediction,
# #             size=(params["height"], params["width"]),
# #             mode="bilinear",
# #             align_corners=False,
# #         )

# #         restored = torch.zeros((*prediction.shape[:-2], original_height, original_width), device=prediction.device, dtype=prediction.dtype)
# #         restored[..., params["top"]:params["top"] + params["height"],  params["left"]:params["left"] + params["width"]] = prediction

# #         if original_ndim == 2:
# #             restored = restored.squeeze(0).squeeze(0)
# #         elif original_ndim == 3:
# #             restored = restored.squeeze(0)

# #         return restored


# class TTARandomApply(torch.nn.Module):
#     """Randomly apply a transform while keeping TTA parameters."""

#     def __init__(self, transform, p=0.5):
#         super().__init__()

#         if not 0.0 <= p <= 1.0:
#             raise ValueError("p must be between 0 and 1")

#         self.transform = transform
#         self.p = p

#     def tta_forward(self, image):
#         apply = torch.rand((), device=image.device).item() < self.p

#         if not apply:
#             return image, {
#                 "apply": False,
#                 "params": None,
#             }

#         if hasattr(self.transform, "tta_forward"):
#             image, params = self.transform.tta_forward(image)
#         else:
#             image = self.transform(image)
#             params = None

#         return image, {
#             "apply": True,
#             "params": params,
#         }

#     def forward(self, image, *masks):
#         apply = torch.rand((), device=image.device).item() < self.p

#         if not apply:
#             return (image, *masks)

#         return self.transform(image, *masks)

#     def inverse(self, prediction, params):
#         if not params["apply"]:
#             return prediction

#         if hasattr(self.transform, "inverse"):
#             return self.transform.inverse(prediction, params["params"])

#         return prediction


# class TTAOneOf(torch.nn.Module):
#     """Randomly select one transform while keeping TTA parameters."""

#     def __init__(self, transforms):
#         super().__init__()

#         if not transforms:
#             raise ValueError("TTAOneOf requires at least one transform")

#         self.transforms = torch.nn.ModuleList(transforms)

#     def tta_forward(self, image):
#         index = torch.randint(
#             len(self.transforms),
#             (),
#             device=image.device,
#         ).item()

#         transform = self.transforms[index]

#         if hasattr(transform, "tta_forward"):
#             image, params = transform.tta_forward(image)
#         else:
#             image = transform(image)
#             params = None

#         return image, {
#             "index": index,
#             "params": params,
#         }

#     def forward(self, image, *masks):
#         index = torch.randint(
#             len(self.transforms),
#             (),
#             device=image.device,
#         ).item()

#         return self.transforms[index]( image, *masks)

#     def inverse(self, prediction, params):
#         transform = self.transforms[params["index"]]

#         if hasattr(transform, "inverse"):
#             return transform.inverse( prediction, params["params"])

#         return prediction


# def build_augmentation(aug):
#     transforms = []

#     def add_with_probability(transform, spec):
#         """Apply a transform only with the probability from the YAML."""

#         p = getattr(spec, "p", 1.0)

#         if not 0.0 <= p <= 1.0:
#             raise ValueError(f"Transform probability must be between 0 and 1, got {p}")

#         if p < 1.0:
#             # transforms.append(v2.RandomApply([transform], p=p))
#             transforms.append(TTARandomApply(transform, p=p))
#         else:
#             transforms.append(transform)

#     for t in aug.transforms:
#         match t.name:
#             # ---------------------------------------------------------
#             # Spatial augmentations
#             # ---------------------------------------------------------

#             case "random_horizontal_flip":
#                 # transforms.append(v2.RandomHorizontalFlip(p=t.p))
#                 transforms.append(TTARandomHorizontalFlip(p=t.p))
#             case "random_vertical_flip":
#                 # transforms.append(v2.RandomVerticalFlip(p=t.p))
#                 transforms.append(TTARandomVerticalFlip(p=t.p))
#             case "random_rotation":
#                 # add_with_probability(v2.RandomRotation(degrees=t.degrees), t)
#                 add_with_probability(TTARandomRotation(degrees=t.degrees), t)
#             case "random_affine":
#                 # add_with_probability(v2.RandomAffine(degrees=t.degrees, translate=t.translate, scale=t.scale, shear=t.shear), t)
#                 add_with_probability(TTARandomAffine(degrees=t.degrees, translate=t.translate, scale=t.scale, shear=t.shear), t)
#             case "random_resized_crop":
#                 # add_with_probability(v2.RandomResizedCrop(size=t.size, scale=t.scale, ratio=t.ratio), t)
#                 add_with_probability(TTARandomResizedCrop( size=t.size, scale=t.scale, ratio=t.ratio), t)
#             case "transpose_axes":
#                 # add_with_probability(TransposeAxes(axes=t.axes), t)
#                 add_with_probability(TTATransposeAxes(axes=t.axes), t)

#             # ---------------------------------------------------------
#             # Intensity augmentations
#             # ---------------------------------------------------------

#             case "color_jitter":
#                 add_with_probability(v2.ColorJitter(brightness=t.brightness, contrast=t.contrast, saturation=t.saturation, hue=t.hue), t)
#             case "contrast":
#                 add_with_probability(v2.RandomAutocontrast(p=1.0), t)
#             case "multiplicative_brightness":
#                 add_with_probability(MultiplicativeBrightness(multiplier=t.multiplier), t)
#             case "brightness_additive":
#                 add_with_probability(AdditiveBrightness(min_value=t.min_value, max_value=t.max_value), t)
#             case "gamma":
#                 add_with_probability(GammaTransform(gamma=t.gamma), t)
#             case "invert":
#                 add_with_probability(InvertImage(), t)
#             case "sharpening":
#                 add_with_probability(v2.RandomAdjustSharpness(sharpness_factor=t.strength, p=1.0), t)

#             # ---------------------------------------------------------
#             # Noise / degradation
#             # ---------------------------------------------------------

#             case "gaussian_blur":
#                 add_with_probability(v2.GaussianBlur(kernel_size=t.kernel_size, sigma=t.sigma), t)
#             case "gaussian_noise":
#                 add_with_probability(AddGaussianNoise(mean=t.mean, sigma=t.sigma), t)
#             case "low_resolution":
#                 add_with_probability(RandomLowResolution(scale=t.scale), t)
#             case "blank_rectangle":
#                 add_with_probability(BlankRectangle(rectangle_size=t.rectangle_size, num_rectangles=t.num_rectangles), t)
#             case "motion_blur":
#                 add_with_probability(MotionBlur(kernel_size=t.kernel_size), t)

#             # ---------------------------------------------------------
#             # 3D-specific augmentations
#             # ---------------------------------------------------------

#             case "smear":
#                 raise NotImplementedError("The 'smear' augmentation is 3D-specific and requires a dedicated implementation.")
#             case "inhomogeneous_slice_illumination":
#                 raise NotImplementedError("The 'inhomogeneous_slice_illumination' augmentation requires a dedicated 3D implementation.")
#             case "brightness_gradient":
#                 raise NotImplementedError("The 'brightness_gradient' augmentation requires a dedicated implementation.")

#             # ---------------------------------------------------------
#             # Cutout
#             # ---------------------------------------------------------

#             case "cutout":
#                 add_with_probability(CoarseDropout(max_holes=t.max_holes, max_size_fraction=t.max_size_fraction, fill=t.fill), t)

#             # ---------------------------------------------------------
#             # OneOf
#             # ---------------------------------------------------------

#             case "oneof":
#                 alternatives = []
#                 for alternative in t.transforms:
#                     alternatives.append(build_augmentation(alternative))
#                 # add_with_probability(OneOf(alternatives), t)
#                 add_with_probability(TTAOneOf(alternatives), t)

#             # ---------------------------------------------------------
#             # Unknown transform
#             # ---------------------------------------------------------

#             case _:
#                 raise ValueError(f"Unknown transform: {t.name}")

#     if not transforms:
#         # return v2.Identity()
#         return TTAIdentity()

#     # return v2.Compose(transforms)
#     return TTAPipeline(transforms)




# import torch
# from torchvision.transforms import v2


# class AddGaussianNoise(torch.nn.Module):
#     def __init__(self, mean=0., sigma=0.05):
#         super().__init__()

#         self.mean = mean
#         self.sigma = sigma

#     def forward(self, image, *masks):
#         noise = torch.randn_like(image)
#         image = image + noise * self.sigma + self.mean
#         image = torch.clamp(image, 0, 1)
#         return (image, *masks)


# def build_augmentation(aug):
#     transforms = []

#     def add_with_probability(transform, spec):
#         """Apply non-flip transforms only with the probability from the YAML."""
#         p = getattr(spec, "p", 1.0)
#         transforms.append(v2.RandomApply([transform], p=p) if p < 1.0 else transform)

#     for t in aug.transforms:
#         match t.name:
#             case "random_horizontal_flip":
#                 transforms.append(v2.RandomHorizontalFlip(p=t.p))
#             case "random_vertical_flip":
#                 transforms.append(v2.RandomVerticalFlip(p=t.p))
#             case "random_rotation":
#                 add_with_probability(v2.RandomRotation(degrees=t.degrees), t)
#             case "random_affine":
#                 add_with_probability(v2.RandomAffine(degrees=t.degrees, translate=t.translate, scale=t.scale, shear=t.shear), t)
#             case "random_resized_crop":
#                 add_with_probability(v2.RandomResizedCrop(size=t.size, scale=t.scale, ratio=t.ratio), t)
#             case "color_jitter":
#                 add_with_probability(v2.ColorJitter(brightness=t.brightness, contrast=t.contrast, saturation=t.saturation, hue=t.hue), t)
#             case "gaussian_blur":
#                 add_with_probability(v2.GaussianBlur(kernel_size=t.kernel_size, sigma=t.sigma), t)
#             case "gaussian_noise":
#                 add_with_probability(AddGaussianNoise(mean=t.mean, sigma=t.sigma), t)
#             case _:
#                 raise ValueError(f"Unknown transform: {t.name}")

#     if not transforms:
#         return v2.Identity()

#     return v2.Compose(transforms)
