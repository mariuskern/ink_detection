import numpy as np
import torch
# from scipy import ndimage
# from skimage.morphology import remove_small_objects
import cv2
# import dask.array as da
# from dask_image.ndmeasure import label as dask_label

# import os
# import shutil
# import tempfile

# import cv2
# import dask.array as da
# import numpy as np
# import torch
# import zarr

# from dask_image.ndmeasure import label


class LabelRefinementPipeline:
    """Sequential pipeline for pseudo-label refinement."""

    def __init__(self, steps, debug=False):
        self.steps = steps
        self.debug = debug

    def __call__(self, predictions):
        is_single = not isinstance(predictions, (list, tuple))

        if is_single:
            predictions = [predictions]
        
        for i, step in enumerate(self.steps):
            predictions = step(predictions)

            if self.debug:
                step_name = f"{i:02d}_{step.__class__.__name__}"

                for prediction in predictions:
                    if not hasattr(prediction, "refinement_debug"):
                        prediction.refinement_debug = {}

                    prediction.refinement_debug[step_name] = {
                        "pseudo_label": prediction.pseudo_label.copy(),
                        "pseudo_mask": prediction.pseudo_mask.copy(),
                        "statistics": self._get_debug_statistics(prediction),
                    }

        if is_single:
            return predictions[0]

        return predictions

    def _get_debug_statistics(self, prediction):
        label = prediction.pseudo_label
        mask = prediction.pseudo_mask

        return {
            "label_pixels": int(label.sum()),
            "valid_pixels": int(mask.sum()),
            "total_pixels": int(mask.size),
            "valid_ratio": float(mask.mean()),
        }


class Threshold(torch.nn.Module):
    """Create pseudo labels and initialize the corresponding mask."""

    def __init__(self, threshold):
        super().__init__()
        self.threshold = threshold

    def forward(self, predictions):
        for prediction in predictions:
            prediction.pseudo_label = (prediction.probability >= self.threshold).astype(np.uint8)
            prediction.pseudo_mask = np.ones_like(prediction.pseudo_label, dtype=bool)
            prediction.pseudo_label_not_refined = prediction.pseudo_label.copy()
            prediction.pseudo_mask_not_refined = prediction.pseudo_mask.copy()

        return predictions


class ConfidenceMask(torch.nn.Module):
    """Mask predictions below a confidence threshold."""

    def __init__(self, threshold, update_label):
        super().__init__()

        self.threshold = threshold
        self.update_label = update_label

    def forward(self, predictions):
        for prediction in predictions:
            mask = prediction.confidence > self.threshold

            if prediction.pseudo_mask is None:
                prediction.pseudo_mask = mask
            else:
                prediction.pseudo_mask &= mask

            if self.update_label:
                prediction.pseudo_label &= mask

        return predictions


class UncertaintyMask(torch.nn.Module):
    """Mask predictions above an uncertainty threshold."""

    def __init__(self, threshold, update_label):
        super().__init__()

        self.threshold = threshold
        self.update_label = update_label

    def forward(self, predictions):
        for prediction in predictions:
            mask = prediction.uncertainty < self.threshold

            if prediction.pseudo_mask is None:
                prediction.pseudo_mask = mask
            else:
                prediction.pseudo_mask &= mask

            if self.update_label:
                prediction.pseudo_label &= mask

        return predictions


class RemoveSmallObjects(torch.nn.Module):
    """
    Remove connected components smaller than min_size.

    Optionally updates the pseudo mask so that removed pixels are also
    excluded from the training mask.
    """

    def __init__(self, min_size, connectivity=4, update_mask=True):
    # def __init__(self, min_size, connectivity=4, update_mask=True, chunk_size=4096):
        super().__init__()

        if min_size < 1:
            raise ValueError("min_size must be at least 1")

        if connectivity not in (4, 8):
            raise ValueError("connectivity must be 4 or 8")

        self.min_size = min_size
        self.connectivity = connectivity
        self.update_mask = update_mask
        # self.chunk_size  = chunk_size 

    def forward(self, predictions):
        for prediction in predictions:
            # label = prediction.pseudo_label.astype(bool)
            label = prediction.pseudo_label.astype(np.uint8)

            # 1

            # refined_label = remove_small_objects(label, min_size=self.max_size, connectivity=self.connectivity)

            # 2

            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(label, connectivity=self.connectivity)

            areas = stats[:, cv2.CC_STAT_AREA]

            keep = areas >= self.min_size
            keep[0] = False  # Hintergrund nicht behalten

            refined_label = keep[labels].astype(np.uint8)

            # 3

            # refined_label = cv2.morphologyEx(
            #     label,
            #     cv2.MORPH_OPEN,
            #     self.kernel,
            #     iterations=self.iterations,
            # )

            # 4

            # mask = prediction.pseudo_label.astype(np.uint8)

            # # Convert large NumPy array into a chunked Dask array.
            # mask_da = da.from_array(mask, chunks=(self.chunk_size, self.chunk_size))

            # # Global connected-component labeling.
            # labels_da, num_labels = dask_label(mask_da, structure=self._structure())

            # # Compute labels.
            # labels = labels_da.compute()

            # # Count pixels per component.
            # component_sizes = np.bincount(labels.ravel())

            # # Keep components >= min_size.
            # keep = component_sizes >= self.min_size
            # keep[0] = False  # background

            # refined_label = keep[labels].astype(np.uint8)

            if self.update_mask:
                removed = (label > 0) & (refined_label == 0)

                prediction.pseudo_mask[removed] = False

            prediction.pseudo_label = refined_label.astype(np.uint8)

        return predictions

    # def _structure(self):
    #     if self.connectivity == 1:
    #         return np.array(
    #             [
    #                 [0, 1, 0],
    #                 [1, 1, 1],
    #                 [0, 1, 0],
    #             ],
    #             dtype=np.uint8,
    #         )

    #     return np.ones((3, 3), dtype=np.uint8)


# class RemoveSmallObjects(torch.nn.Module):
#     """
#     Remove connected components smaller than min_size using
#     chunked connected-component labeling.

#     The global component labels are stored on disk as a Zarr array,
#     so the complete label image never has to fit into RAM.

#     Parameters
#     ----------
#     min_size : int
#         Minimum number of pixels a connected component must contain.
#     connectivity : int
#         1 for 4-connectivity, 2 for 8-connectivity.
#     chunk_size : int
#         Spatial chunk size used for Dask processing.
#     update_mask : bool
#         If True, pixels removed from the pseudo label are also removed
#         from the pseudo mask.
#     tmp_dir : str or None
#         Directory used for the temporary Zarr label array.
#         If None, the system temporary directory is used.
#     """

#     def __init__(
#         self,
#         min_size=20_000,
#         connectivity=1,
#         chunk_size=4000,
#         update_mask=True,
#         tmp_dir=None,
#     ):
#         super().__init__()

#         if min_size < 1:
#             raise ValueError("min_size must be at least 1")

#         if connectivity not in (1, 2):
#             raise ValueError("connectivity must be 1 or 2")

#         if chunk_size < 1:
#             raise ValueError("chunk_size must be at least 1")

#         self.min_size = min_size
#         self.connectivity = connectivity
#         self.chunk_size = chunk_size
#         self.update_mask = update_mask
#         self.tmp_dir = tmp_dir

#     def forward(self, predictions):
#         for prediction in predictions:
#             prediction.pseudo_label = self._process(
#                 prediction.pseudo_label,
#                 prediction.pseudo_mask if self.update_mask else None,
#             )

#         return predictions

#     def _process(self, pseudo_label, pseudo_mask):
#         # ------------------------------------------------------------
#         # 1. Create Dask array
#         # ------------------------------------------------------------

#         mask = da.from_array(
#             pseudo_label,
#             chunks=(self.chunk_size, self.chunk_size),
#         )

#         structure = self._get_structure()

#         # ------------------------------------------------------------
#         # 2. Global connected components
#         # ------------------------------------------------------------

#         labels, num_labels = label(
#             mask,
#             structure=structure,
#         )

#         print(
#             f"Connected components: {num_labels.compute()}"
#         )

#         # ------------------------------------------------------------
#         # 3. Store global labels on disk
#         # ------------------------------------------------------------

#         os.makedirs(self.tmp_dir, exist_ok=True)

#         temp_root = tempfile.mkdtemp(
#             dir=self.tmp_dir,
#             prefix="remove_small_objects_",
#         )

#         labels_path = os.path.join(
#             temp_root,
#             "labels.zarr",
#         )

#         try:
#             print("Computing connected components...")

#             da.to_zarr(
#                 labels,
#                 labels_path,
#                 overwrite=True,
#             )

#             # Free the Dask graph.
#             del labels

#             # --------------------------------------------------------
#             # 4. Determine global component sizes
#             # --------------------------------------------------------

#             labels_disk = da.from_zarr(labels_path)

#             component_sizes = self._get_component_sizes(
#                 labels_disk,
#                 num_labels,
#             )

#             # --------------------------------------------------------
#             # 5. Determine components to keep
#             # --------------------------------------------------------

#             keep = component_sizes >= self.min_size

#             # Component 0 is background.
#             keep[0] = False

#             print(
#                 f"Components kept: {np.count_nonzero(keep)} "
#                 f"/ {len(keep) - 1}"
#             )

#             # --------------------------------------------------------
#             # 6. Create refined mask chunk-by-chunk
#             # --------------------------------------------------------

#             refined = labels_disk.map_blocks(
#                 self._filter_labels,
#                 keep=keep,
#                 dtype=np.uint8,
#             )

#             refined = refined.compute()

#             # --------------------------------------------------------
#             # 7. Update pseudo mask
#             # --------------------------------------------------------

#             if pseudo_mask is not None:
#                 removed = (
#                     (pseudo_label > 0)
#                     & (refined == 0)
#                 )

#                 pseudo_mask[removed] = False

#             return refined

#         finally:
#             # --------------------------------------------------------
#             # 8. Delete temporary label store
#             # --------------------------------------------------------

#             shutil.rmtree(
#                 temp_root,
#                 ignore_errors=True,
#             )

#     def _get_component_sizes(self, labels, num_labels):
#         delayed_chunks = labels.to_delayed().ravel()

#         # Determine the maximum label ID first
#         max_label = 0

#         for delayed_chunk in delayed_chunks:
#             chunk = delayed_chunk.compute()

#             if chunk.size > 0:
#                 max_label = max(max_label, int(chunk.max()))

#             del chunk

#         sizes = np.zeros(max_label + 1, dtype=np.int64)

#         for i, delayed_chunk in enumerate(delayed_chunks):
#             print(
#                 f"Counting components "
#                 f"{i + 1}/{len(delayed_chunks)}...",
#                 end="\r",
#             )

#             chunk = delayed_chunk.compute()

#             ids, counts = np.unique(
#                 chunk,
#                 return_counts=True,
#             )

#             sizes[ids] += counts

#             del chunk

#         print()

#         return sizes

#     @staticmethod
#     def _filter_labels(block, keep):
#         """
#         Convert global component labels into a binary mask.
#         """

#         return keep[block].astype(np.uint8)

#     def _get_structure(self):
#         if self.connectivity == 1:
#             return np.array(
#                 [
#                     [0, 1, 0],
#                     [1, 1, 1],
#                     [0, 1, 0],
#                 ],
#                 dtype=np.uint8,
#             )

#         return np.ones(
#             (3, 3),
#             dtype=np.uint8,
#         )


class BinaryClosing(torch.nn.Module):
    """
    Apply morphological binary closing to the pseudo label.

    Optionally restricts closing to regions where the prediction probability
    exceeds a given threshold.

    Newly created pixels can optionally be added to the pseudo mask.
    """

    def __init__(self, kernel_size=3, iterations=1, probability_threshold=None, update_mask=False):
        super().__init__()

        if kernel_size < 1 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer")

        if iterations < 1:
            raise ValueError("iterations must be at least 1")

        if probability_threshold is not None and (not 0.0 <= probability_threshold <= 1.0):
            raise ValueError("probability_threshold must be between 0 and 1")

        self.kernel_size = kernel_size
        self.iterations = iterations
        self.probability_threshold = probability_threshold
        self.update_mask = update_mask

        self.kernel = np.ones((self.kernel_size, self.kernel_size), dtype=np.uint8)

    def forward(self, predictions):
        # structure = np.ones((self.kernel_size, self.kernel_size), dtype=bool)

        for prediction in predictions:
            # label = prediction.pseudo_label.astype(bool)
            label = prediction.pseudo_label.astype(np.uint8)

            # if self.probability_threshold is not None:
            #     probability_mask = prediction.probability >= self.probability_threshold
            #     closing_input = label & probability_mask
            # else:
            #     closing_input = label

            # refined_label = ndimage.binary_closing(closing_input, structure=structure, iterations=self.iterations)

            # if self.probability_threshold is not None:
            #     refined_label = label & ~probability_mask | refined_label & probability_mask

            # closed = ndimage.binary_closing(label, structure=structure, iterations=self.iterations)

            closed = cv2.morphologyEx(
                label,
                cv2.MORPH_CLOSE,
                self.kernel,
                iterations=self.iterations,
            )

            if self.probability_threshold is not None:
                refined_label = closed & (prediction.probability >= self.probability_threshold)
            else:
                refined_label = closed

            if self.update_mask:
                added = refined_label & ~label
                prediction.pseudo_mask[added] = True

            prediction.pseudo_label = refined_label.astype(np.uint8)

        return predictions


def build_label_refinement(cfg):
    steps = []

    for step in cfg.steps:
        match step.name:
            case "threshold":
                steps.append(
                    Threshold(
                        threshold=step.threshold,
                    )
                )

            case "confidence_mask":
                steps.append(
                    ConfidenceMask(
                        threshold=step.threshold,
                        update_label=step.update_label,
                    )
                )
            case "uncertainty_mask":
                steps.append(
                    UncertaintyMask(
                        threshold=step.threshold,
                        update_label=step.update_label,
                    )
                )
            case "remove_small_objects":
                steps.append(
                    RemoveSmallObjects(
                        min_size=step.min_size,
                        connectivity=step.connectivity,
                        update_mask=step.update_mask,
                        # chunk_size=step.chunk_size,
                        # tmp_dir=step.tmp_dir
                    )
                )

            case "binary_closing":
                steps.append(
                    BinaryClosing(
                        kernel_size=step.kernel_size,
                        iterations=step.iterations,
                        probability_threshold=step.probability_threshold,
                        update_mask=step.update_mask,
                    )
                )
            case _:
                raise ValueError(
                    f"Unknown label refinement step: {step.name}"
                )

    return LabelRefinementPipeline(steps, debug=cfg.debug)
