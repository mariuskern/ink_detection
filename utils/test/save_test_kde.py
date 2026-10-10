import os
import sys
from pathlib import Path

import numpy as np

from dotenv import load_dotenv

load_dotenv()

sys.path.append(os.getenv("CODE_DIR"))

from utils.save import _save_kde_plot


# Testbild mit Werten zwischen 0 und 1 erzeugen
rng = np.random.default_rng(42)
image = rng.uniform(0, 1, size=(2048, 2048)).astype(np.float32)

print(f"min={image.min():.4f}, max={image.max():.4f}, mean={image.mean():.4f}")

# Visualisierung mit deiner Methode
_save_kde_plot(
    output_dir=Path("./outputs"),
    name="test",
    segment="synthetic",
    image=image,
    name_suffix=None,
    create_subfolders=False,
    xlabel="Pixel value"
)