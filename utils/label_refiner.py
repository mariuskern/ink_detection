import numpy as np

from scipy.ndimage import gaussian_filter
from skimage.filters import sobel
from skimage.morphology import (
    binary_opening,
    binary_closing,
    disk,
    remove_small_objects,
    reconstruction,
)
from skimage.measure import label, regionprops


class LabelRefiner:

    def __init__(
        self,
        strong_threshold=0.80,
        weak_threshold=0.30,
        confidence_threshold=0.70,
        uncertainty_threshold=0.20,
        # use_confidence_mask=True,
        # use_uncertainty_mask=True,
        min_component_size=20,
        opening_radius=1,
        closing_radius=2,
        local_window_sigma=3.0,
        use_opening=True,
        use_closing=True,
        use_reconstruction=True,
        use_components=True,
        use_local_mean=True,
        # use_gradient=True,
    ):
        self.strong_threshold = strong_threshold
        self.weak_threshold = weak_threshold
        self.confidence_threshold = confidence_threshold
        self.uncertainty_threshold = uncertainty_threshold
        # self.use_confidence_mask = use_confidence_mask,
        # self.use_uncertainty_mask = use_uncertainty_mask,
        self.min_component_size = min_component_size
        self.opening_radius = opening_radius
        self.closing_radius = closing_radius
        self.local_window_sigma = local_window_sigma
        self.use_opening = use_opening
        self.use_closing = use_closing
        self.use_reconstruction = use_reconstruction
        self.use_components = use_components
        self.use_local_mean = use_local_mean
        # self.use_gradient = use_gradient

    def refine(self, prediction):
        """
        Refine one Prediction object.

        Returns:
            refined_label:
                0 = background
                1 = ink

            refined_mask:
                True = pixel can be used for training
                False = ignore pixel
        """

        prob = prediction.probability.astype(np.float32)
        confidence = prediction.confidence
        uncertainty = prediction.uncertainty

        # ---------------------------------------------------------
        # 1. Strong and weak predictions
        # ---------------------------------------------------------

        strong = prob >= self.strong_threshold
        weak = prob >= self.weak_threshold

        label_not_refined = strong.copy()

        # ---------------------------------------------------------
        # 2. Confidence / uncertainty filtering
        # ---------------------------------------------------------

        reliable = np.ones_like(prob, dtype=bool)

        if confidence is not None:
            reliable &= confidence >= self.confidence_threshold

        if uncertainty is not None:
            reliable &= uncertainty <= self.uncertainty_threshold

        mask_not_refined = reliable.copy()

        # strong &= reliable

        # ---------------------------------------------------------
        # 3. Morphological cleanup of strong seeds
        # ---------------------------------------------------------

        seeds = strong.copy()

        if self.use_components:
            seeds = remove_small_objects(seeds, min_size=self.min_component_size)

        if self.use_opening:
            structure = disk(self.opening_radius)
            seeds = binary_opening(seeds, structure)

        # ---------------------------------------------------------
        # 4. Morphological reconstruction
        # ---------------------------------------------------------

        if self.use_reconstruction:
            # Important:
            # reconstruction may only happen inside the weak
            # teacher-supported prediction.

            marker = seeds.astype(np.uint8)
            mask = weak.astype(np.uint8)

            reconstructed = reconstruction(marker, mask, method="dilation")
            refined = reconstructed.astype(bool)
        else:
            refined = seeds.copy()

        # ---------------------------------------------------------
        # 5. Closing
        # ---------------------------------------------------------

        if self.use_closing:
            structure = disk(self.closing_radius)
            refined = binary_closing(refined, structure)

            # Never leave the teacher-supported region
            refined &= weak

        # ---------------------------------------------------------
        # 6. Local probability consistency
        # ---------------------------------------------------------

        if self.use_local_mean:
            local_mean = gaussian_filter(prob, sigma=self.local_window_sigma)
            # Pixels should not be dramatically less probable
            # than their local neighbourhood.
            local_consistent = local_mean >= self.weak_threshold
            refined &= local_consistent

        # # ---------------------------------------------------------
        # # 7. Gradient information
        # # ---------------------------------------------------------

        # gradient = None

        # if self.use_gradient:
        #     gradient = sobel(prob)

        #     # Normalize gradient
        #     gradient = gradient / (gradient.max() + 1e-8)

        #     # Gradient is mainly useful for boundary information.
        #     #
        #     # We do NOT simply say:
        #     # high gradient = ink
        #     #
        #     # Instead, keep refined regions supported by probability.
        #     #
        #     # Boundary information is stored for later experiments.

        # ---------------------------------------------------------
        # 8. Build final training mask
        # ---------------------------------------------------------

        refined_label = refined.astype(np.uint8)

        refined_mask = reliable.copy()
        changed = refined_label != label_not_refined
        refined_mask &= ~changed

        # refined_mask = np.ones_like(refined_label, dtype=bool)

        # # High uncertainty -> ignore
        # if uncertainty is not None:
        #     refined_mask &= (uncertainty <= self.uncertainty_threshold)

        # # Very ambiguous probabilities -> ignore
        # ambiguous = ((prob > 0.45) & (prob < 0.55))
        # refined_mask &= ~ambiguous

        # ---------------------------------------------------------
        # 9. Save debug information
        # ---------------------------------------------------------

        debug = {
            "strong": strong,
            "weak": weak,
            "reliable": reliable,
            "seeds": seeds,
            "refined": refined,
            "changed": changed,
            "added": refined & ~label_not_refined,
            "removed": label_not_refined & ~refined,
            # "gradient": gradient,
        }

        return label_not_refined, mask_not_refined, refined_label, refined_mask, debug