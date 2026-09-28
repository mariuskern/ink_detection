import os
import sys
from pathlib import Path

from PIL import Image
import numpy as np
import hydra
import cv2

Image.MAX_IMAGE_PIXELS = None

from dotenv import load_dotenv

load_dotenv()

sys.path.append(os.getenv("CODE_DIR"))

from utils import Prediction
from utils.refinement import build_label_refinement


PREDICTION_PATH = Path("C:/Users/mariu/Documents/Studium/Masterarbeit/label_refinement_test_data")

PROBABILITY_PATH = PREDICTION_PATH / "prediction"
CONFIDENCE_PATH = None # PREDICTION_PATH / "confidence"
UNCERTAINTY_PATH = None # PREDICTION_PATH / "uncertainty"


def load_image(path):
    image = np.array(Image.open(path))

    if image.ndim == 3:
        image = image[..., 0]

    if image.dtype == np.uint8:
        image = image.astype(np.float32) / 255.0
    else:
        image = image.astype(np.float32)

    return image


def load_predictions(probability_dir, confidence_dir=None, uncertainty_dir=None):
    probability_dir = Path(probability_dir)

    extensions = {
        ".png",
        ".jpg",
        ".jpeg",
        ".tif",
        ".tiff",
    }

    # probability_files = sorted(p for p in probability_dir.iterdir() if p.is_file() and p.suffix.lower() in extensions and p.stem.endswith("_prediction"))
    probability_files = sorted(p for p in probability_dir.iterdir() if p.is_file() and p.suffix.lower() in extensions)

    if not probability_files:
        raise FileNotFoundError(f"Keine Prediction-Bilder gefunden in {probability_dir}")

    results = []

    for probability_path in probability_files:
        segment = probability_path.stem.removesuffix("_prediction")

        print(f"Loading segment: {segment}")

        probability = load_image(probability_path)

        confidence = None

        if confidence_dir is not None:
            confidence_path = Path(confidence_dir) / f"{segment}_confidence{probability_path.suffix}"

            if not confidence_path.exists():
                raise FileNotFoundError(
                    f"Confidence für '{segment}' nicht gefunden:\n"
                    f"{confidence_path}"
                )

            confidence = load_image(confidence_path)
        else:
            confidence = np.ones_like(probability)

        uncertainty = None

        if uncertainty_dir is not None:
            uncertainty_path = Path(uncertainty_dir) / f"{segment}_uncertainty{probability_path.suffix}"

            if not uncertainty_path.exists():
                raise FileNotFoundError(
                    f"Uncertainty für '{segment}' nicht gefunden:\n"
                    f"{uncertainty_path}"
                )

            uncertainty = load_image(uncertainty_path)
        else:
            uncertainty = np.zeros_like(probability)

        prediction = Prediction(
            segment=segment,
            probability=probability,
            confidence=confidence,
            uncertainty=uncertainty,
        )

        results.append(prediction)

    print(f"\nLoaded {len(results)} segments.")

    return results


def resize_for_plot(array, max_size=1500, interpolation=cv2.INTER_AREA):
    array = array.astype(np.uint8)

    h, w = array.shape[-2:]

    scale = min(max_size / h, max_size / w)

    if scale >= 1:
        return array

    new_h = int(h * scale)
    new_w = int(w * scale)

    return cv2.resize(array, (new_w, new_h), interpolation=interpolation)


def plot_refinement(prediction):
    import matplotlib
    matplotlib.use("TkAgg", force=True)
    import matplotlib.pyplot as plt

    probability = prediction.probability
    confidence = prediction.confidence
    uncertainty = prediction.uncertainty
    debug = getattr(prediction, "refinement_debug", {})

    if not debug:
        raise ValueError("Keine Refinement-Debug-Daten vorhanden. Aktiviere debug=True in der Refinement-Pipeline.")

    debug_items = list(debug.items())

    max_steps_per_block = 5

    # In Blöcke mit jeweils maximal 5 Schritten aufteilen
    blocks = [debug_items[i:i + max_steps_per_block] for i in range(0, len(debug_items), max_steps_per_block)]

    n_blocks = len(blocks)

    # 5 Bilder pro Schritt:
    # Prediction | Label | Mask | Label ∩ Mask | Change
    n_cols = 1 + n_blocks * 4
    n_rows = max_steps_per_block

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows), squeeze=False)

    fig.suptitle(f"Label Refinement - {prediction.segment}", fontsize=18,)

    # Alles zunächst deaktivieren
    for ax in axes.flat:
        ax.axis("off")

    # # ---------------------------------------------------------
    # # Original Prediction
    # # ---------------------------------------------------------

    # axes[0, 0].imshow(probability, cmap="gray", vmin=0, vmax=1)
    # axes[0, 0].set_title("Original Prediction Probability", fontsize=11)

    # ---------------------------------------------------------
    # Refinement-Schritte
    # ---------------------------------------------------------

    for block_idx, block in enumerate(blocks):
        col_offset = 1 + block_idx * 4

        for row, (step_name, state) in enumerate(block):
            global_idx = block_idx * max_steps_per_block + row

            pseudo_label = state["pseudo_label"]
            pseudo_mask = state["pseudo_mask"]

            # Label vor diesem Schritt
            if global_idx == 0:
                previous_label = probability >= 0.5
            else:
                previous_label = debug_items[global_idx - 1][1]["pseudo_label"]

            # Änderungen dieses Schrittes
            difference = np.logical_xor(previous_label, pseudo_label)

            # -------------------------------------------------
            # Prediction
            # -------------------------------------------------

            axes[row, col_offset - 1].imshow(resize_for_plot(probability), cmap="gray", vmin=0, vmax=1)
            axes[row, col_offset - 1].set_title("Original Prediction Probability", fontsize=11)

            # -------------------------------------------------
            # Label
            # -------------------------------------------------

            axes[row, col_offset].imshow(resize_for_plot(pseudo_label), cmap="gray", vmin=0, vmax=1)
            axes[row, col_offset].set_title(f"{global_idx + 1}: {step_name}\nLabel", fontsize=11)

            # -------------------------------------------------
            # Training Mask
            # -------------------------------------------------

            axes[row, col_offset + 1].imshow(resize_for_plot(pseudo_mask), cmap="gray", vmin=0, vmax=1)
            axes[row, col_offset + 1].set_title(f"{global_idx + 1}: {step_name}\nTraining Mask", fontsize=11)

            # -------------------------------------------------
            # Label ∩ Mask
            # -------------------------------------------------

            masked_label = np.where(pseudo_mask, pseudo_label, 0)

            axes[row, col_offset + 2].imshow(resize_for_plot(masked_label), cmap="gray", vmin=0, vmax=1)
            axes[row, col_offset + 2].set_title(f"{global_idx + 1}: {step_name}\nLabel ∩ Mask", fontsize=11)

            # -------------------------------------------------
            # Label Change
            # -------------------------------------------------

            axes[row, col_offset + 3].imshow(resize_for_plot(difference), cmap="gray", vmin=0, vmax=1)
            axes[row, col_offset + 3].set_title(f"{global_idx + 1}: {step_name}\nLabel Change", fontsize=11)

    plt.tight_layout(rect=(0, 0, 1, 0.96))
    plt.show()


@hydra.main( version_base=None, config_path="../../conf/refinement", config_name="debug")
def main(cfg):

    # ---------------------------------------------------------
    # Load predictions
    # ---------------------------------------------------------

    predictions = load_predictions(
        probability_dir=PROBABILITY_PATH,
        confidence_dir=CONFIDENCE_PATH,
        uncertainty_dir=UNCERTAINTY_PATH,
    )

    # ---------------------------------------------------------
    # Build refinement pipeline
    # ---------------------------------------------------------

    refinement = build_label_refinement(cfg)

    # ---------------------------------------------------------
    # Refine
    # ---------------------------------------------------------

    predictions = refinement(predictions)

    # ---------------------------------------------------------
    # Plot
    # ---------------------------------------------------------

    for i, prediction in enumerate(predictions):
        print(f"\nProcessing segment {i + 1}/{len(predictions)}: {prediction.segment}")

        plot_refinement(prediction)


if __name__ == "__main__":
    main()





# import os
# import sys
# from pathlib import Path
# from PIL import Image

# import numpy as np
# import hydra
# import matplotlib.pyplot as plt

# from dotenv import load_dotenv

# load_dotenv()

# sys.path.append(os.getenv("CODE_DIR"))

# from utils import Prediction, postprocess_predictions

# PREDICTION_PATH = Path("C:/Users/mariu/Documents/Studium/Masterarbeit/label_refinement_test_data")

# PROBABILITY_PATH = PREDICTION_PATH / "prediction"
# CONFIDENCE_PATH = PREDICTION_PATH / "confidence"
# UNCERTAINTY_PATH = PREDICTION_PATH / "uncertainty"


# def load_image(path):
#     image = np.array(Image.open(path))

#     if image.ndim == 3:
#         image = image[..., 0]

#     if image.dtype == np.uint8:
#         image = image.astype(np.float32) / 255.0
#     else:
#         image = image.astype(np.float32)

#     return image


# def load_predictions(probability_dir, confidence_dir=None, uncertainty_dir=None):
#     probability_dir = Path(probability_dir)

#     extensions = {
#         ".png",
#         ".jpg",
#         ".jpeg",
#         ".tif",
#         ".tiff",
#     }

#     probability_files = sorted(
#         p
#         for p in probability_dir.iterdir()
#         if p.is_file()
#         and p.suffix.lower() in extensions
#         and p.stem.endswith("_prediction")
#     )

#     if not probability_files:
#         raise FileNotFoundError(
#             f"Keine Prediction-Bilder gefunden in {probability_dir}"
#         )

#     results = []

#     for probability_path in probability_files:
#         segment = probability_path.stem.removesuffix("_prediction")

#         print(f"Loading segment: {segment}")

#         probability = load_image(probability_path)

#         confidence = None
#         if confidence_dir is not None:
#             confidence_path = (
#                 Path(confidence_dir) / f"{segment}_confidence{probability_path.suffix}"
#             )

#             if not confidence_path.exists():
#                 raise FileNotFoundError(
#                     f"Confidence für '{segment}' nicht gefunden:\n{confidence_path}"
#                 )

#             confidence = load_image(confidence_path)

#         uncertainty = None
#         if uncertainty_dir is not None:
#             uncertainty_path = (
#                 Path(uncertainty_dir)
#                 / f"{segment}_uncertainty{probability_path.suffix}"
#             )

#             if not uncertainty_path.exists():
#                 raise FileNotFoundError(
#                     f"Uncertainty für '{segment}' nicht gefunden:\n{uncertainty_path}"
#                 )

#             uncertainty = load_image(uncertainty_path)

#         prediction = Prediction(
#             segment=segment,
#             probability=probability,
#             confidence=confidence,
#             uncertainty=uncertainty,
#         )

#         results.append(prediction)

#     print(f"\nLoaded {len(results)} segments.")

#     return results


# def plot_refinement(prediction):
#     probability = prediction.probability
#     confidence = prediction.confidence
#     uncertainty = prediction.uncertainty

#     pseudo_label = prediction.pseudo_label
#     pseudo_mask = prediction.pseudo_mask

#     refined_label = prediction.refined_label
#     refined_mask = prediction.refined_mask

#     debug = prediction.debug

#     strong = debug.get("strong")
#     weak = debug.get("weak")
#     reliable = debug.get("reliable")
#     seeds = debug.get("seeds")
#     refined = debug.get("refined")
#     gradient = debug.get("gradient")

#     fig, axes = plt.subplots(3, 4, figsize=(16, 12))

#     fig.suptitle(f"Label Refinement - {prediction.segment}", fontsize=16)

#     axes[0, 0].imshow(probability, cmap="gray", vmin=0, vmax=1)
#     axes[0, 0].set_title("Probability")

#     if confidence is not None:
#         axes[0, 1].imshow(confidence, cmap="gray", vmin=0, vmax=1,)
#         axes[0, 1].set_title("Confidence")
#     else:
#         axes[0, 1].set_title("Confidence: None")

#     if uncertainty is not None:
#         axes[0, 2].imshow(uncertainty, cmap="gray")
#         axes[0, 2].set_title("Uncertainty")
#     else:
#         axes[0, 2].set_title("Uncertainty: None")

#     axes[0, 3].imshow(pseudo_label, cmap="gray", vmin=0, vmax=1)
#     axes[0, 3].set_title("Original Label")

#     if strong is not None:
#         axes[1, 0].imshow(strong, cmap="gray")
#         axes[1, 0].set_title("Strong")
#     else:
#         axes[1, 0].set_title("Strong: None")

#     if weak is not None:
#         axes[1, 1].imshow(weak, cmap="gray")
#         axes[1, 1].set_title("Weak")
#     else:
#         axes[1, 1].set_title("Weak: None")

#     if reliable is not None:
#         axes[1, 2].imshow(reliable, cmap="gray")
#         axes[1, 2].set_title("Reliable")
#     else:
#         axes[1, 2].set_title("Reliable: None")

#     if seeds is not None:
#         axes[1, 3].imshow(seeds, cmap="gray")
#         axes[1, 3].set_title("Seeds")
#     else:
#         axes[1, 3].set_title("Seeds: None")

#     if refined is not None:
#         axes[2, 0].imshow(refined, cmap="gray")
#         axes[2, 0].set_title("Refined")
#     else:
#         axes[2, 0].set_title("Refined: None")

#     if gradient is not None:
#         axes[2, 1].imshow(gradient, cmap="gray")
#         axes[2, 1].set_title("Gradient")
#     else:
#         axes[2, 1].set_title("Gradient: None")

#     axes[2, 2].imshow(refined_label, cmap="gray", vmin=0, vmax=1)
#     axes[2, 2].set_title("Final Label")

#     axes[2, 3].imshow(refined_mask, cmap="gray")
#     axes[2, 3].set_title("Final Training Mask")

#     for ax in axes.flat:
#         ax.axis("off")

#     plt.tight_layout()
#     plt.show()


# @hydra.main(
#     version_base=None,
#     config_path="../../conf",
#     config_name="config",
# )
# def main(cfg):
#     results = load_predictions(probability_dir=PROBABILITY_PATH, confidence_dir=CONFIDENCE_PATH, uncertainty_dir=UNCERTAINTY_PATH)
#     results = postprocess_predictions(
#         results,
#         strong_threshold=cfg.train.refinement.strong_threshold,
#         weak_threshold=cfg.train.refinement.weak_threshold,
#         confidence_threshold=cfg.train.refinement.confidence_threshold,
#         uncertainty_threshold=cfg.train.refinement.uncertainty_threshold,
#         # use_confidence_mask=cfg.train.refinement.use_confidence_mask,
#         # use_uncertainty_mask=cfg.train.refinement.use_uncertainty_mask,
#         min_component_size=cfg.train.refinement.min_component_size,
#         opening_radius=cfg.train.refinement.opening_radius,
#         closing_radius=cfg.train.refinement.closing_radius,
#         local_window_sigma=cfg.train.refinement.local_window_sigma,
#         use_opening=cfg.train.refinement.use_opening,
#         use_closing=cfg.train.refinement.use_closing,
#         use_reconstruction=cfg.train.refinement.use_reconstruction,
#         use_components=cfg.train.refinement.use_components,
#         use_local_mean=cfg.train.refinement.use_local_mean,
#     )

#     for i, prediction in enumerate(results):
#         print(f"\nProcessing segment " f"{i + 1}/{len(results)}: " f"{prediction.segment}")

#         plot_refinement(prediction)


# if __name__ == "__main__":
#     main()
