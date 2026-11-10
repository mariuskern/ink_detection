#!/usr/bin/env python3
"""Plot W&B validation Dice by ablation ID and noise, averaging seeds first.

Usage:
    python plot_wandb_dice.py wandb_export.csv
    python plot_wandb_dice.py wandb_export.csv --epochs-per-iteration 50 --save dice.png

Requires: pandas, numpy, matplotlib
"""

import argparse
import colorsys
import re
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch


# ============================================================================
# Constants
# ============================================================================

# Example: a4_high_seed_4042_20261108_... - metrics_val/dice
DICE_COLUMN_RE = re.compile(
    r"^(?P<id>a\d+)_(?P<noise>low|medium|high)_seed_(?P<seed>\d+)_.* - metrics_val/dice$"
)
NOISE_ORDER = ("low", "medium", "high")
NOISE_LIGHTNESS = {"low": 0.75, "medium": 0.50, "high": 0.30}


# ============================================================================
# Read and aggregate metrics
# ============================================================================

def read_grouped_metrics(csv_path, epochs_per_iteration):
    # ------------------------------------------------------------------------
    # CSV input / validation
    # ------------------------------------------------------------------------

    df = pd.read_csv(csv_path)
    step_column = "steps/metrics_val_dice_step"

    if step_column not in df:
        raise ValueError(f"Spalte {step_column!r} fehlt im CSV-Export.")

    steps = pd.to_numeric(df[step_column], errors="coerce")

    if steps.isna().any() or steps.duplicated().any():
        raise ValueError("Die Epochen-/Schrittspalte enthält fehlende oder doppelte Werte.")

    if not np.allclose(steps, steps.round()):
        raise ValueError("Die Schrittspalte muss ganzzahlige Epochenindizes enthalten.")

    # Index by the validation step rather than trusting the row order.
    df.index = steps.astype(int)
    df = df.sort_index()

    # ------------------------------------------------------------------------
    # Group runs by ablation ID and noise
    # ------------------------------------------------------------------------

    grouped = defaultdict(dict)

    for col in df.columns:
        match = DICE_COLUMN_RE.fullmatch(col)

        if match is None:
            continue  # Ignore __MIN, __MAX, _step etc.

        ablation_id = match.group("id")
        noise = match.group("noise")
        seed = int(match.group("seed"))
        key = (ablation_id, noise)

        if seed in grouped[key]:
            raise ValueError(f"Seed {seed} kommt für {key} mehrfach vor.")

        grouped[key][seed] = pd.to_numeric(df[col], errors="coerce")

    if not grouped:
        raise ValueError("Keine passenden Dice-Spalten gefunden. CSV-Spaltennamen prüfen.")

    # ------------------------------------------------------------------------
    # Average seeds and select iteration maxima
    # ------------------------------------------------------------------------

    results = {}

    for key, seed_series in sorted(
        grouped.items(), key=lambda kv: (int(kv[0][0][1:]), NOISE_ORDER.index(kv[0][1]))
    ):
        run_df = pd.DataFrame(seed_series)

        # Average Dice across seeds at *the same epoch*.
        # Only average epochs for which ALL seeds have values.
        mean_dice = run_df.mean(axis=1).where(run_df.notna().all(axis=1))

        iteration_max = {}
        iterations = sorted(set(int(step // epochs_per_iteration) for step in run_df.index if step >= 0))

        for idx in iterations:
            expected_steps = np.arange(
                idx * epochs_per_iteration, (idx + 1) * epochs_per_iteration
            )
            block = mean_dice.reindex(expected_steps)

            if block.isna().any():
                print(
                    f"WARNUNG: {key[0]} / {key[1]}: Iteration {idx + 1} "
                    f"übersprungen ({block.notna().sum()}/{epochs_per_iteration} "
                    "vollständige Epochen über alle Seeds)."
                )
                continue

            iteration_max[idx + 1] = float(block.max())

        if not iteration_max:
            print(f"WARNUNG: Keine vollständige Iteration für {key[0]} / {key[1]}.")
            continue

        results[key] = {
            "per_iteration": iteration_max,
            "best_overall": max(iteration_max.values()),
            "seeds": sorted(seed_series),
        }

    if not results:
        raise ValueError(
            "Keine vollständigen Iterationen gefunden. Prüfe, ob der CSV-Export "
            "alle 50 Validation-Epochen je Iteration enthält."
        )

    return results


# ============================================================================
# Plot results
# ============================================================================

def plot_results(results, save_path=None):
    # ------------------------------------------------------------------------
    # Figure / colors / line styles
    # ------------------------------------------------------------------------

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), constrained_layout=True)
    ablations = sorted({k[0] for k in results}, key=lambda x: int(x[1:]))

    palette = plt.get_cmap("tab10" if len(ablations) <= 10 else "hsv")
    colors = {}

    for index, ablation_id in enumerate(ablations):
        base_color = palette(index if len(ablations) <= 10 else index / len(ablations))
        hue, _, saturation = colorsys.rgb_to_hls(*base_color[:3])

        for noise, lightness in NOISE_LIGHTNESS.items():
            colors[(ablation_id, noise)] = colorsys.hls_to_rgb(hue, lightness, saturation)

    line_styles = {
        name: style
        for name, style in zip(ablations, ["-", "--", ":", "-.", (0, (3, 1, 1, 1))])
    }

    # ------------------------------------------------------------------------
    # Best mean Dice per NST iteration
    # ------------------------------------------------------------------------

    for (ablation_id, noise), result in results.items():
        per_iteration = result["per_iteration"]

        ax1.plot(
            list(per_iteration),
            list(per_iteration.values()),
            label=f"{ablation_id} / {noise} (n={len(result['seeds'])})",
            color=colors[(ablation_id, noise)],
            linestyle=line_styles.get(ablation_id, "-"),
            marker="o",
            markersize=4,
        )

    ax1.set_title("Best mean Dice per NST iteration")
    ax1.set_xlabel("NST iteration")
    ax1.set_ylabel("Dice")
    ax1.set_xticks(sorted({i for r in results.values() for i in r["per_iteration"]}))
    ax1.grid(alpha=0.3)
    ax1.legend(fontsize=8, loc="best", ncol=2)

    # ------------------------------------------------------------------------
    # Best mean Dice across all NST iterations
    # ------------------------------------------------------------------------

    group_width = 0.8
    bar_width = group_width / len(NOISE_ORDER)
    xs = np.arange(len(ablations))

    for offset, noise in enumerate(NOISE_ORDER):
        heights = [
            results[(a, noise)]["best_overall"] if (a, noise) in results else np.nan
            for a in ablations
        ]

        ax2.bar(
            xs - group_width / 2 + (offset + 0.5) * bar_width,
            heights,
            width=bar_width,
            label=noise,
            color=[colors[(a, noise)] for a in ablations],
        )

    ax2.set_xticks(xs, ablations)
    ax2.set_title("Best mean Dice across all NST iterations")
    ax2.set_xlabel("Ablation ID")
    ax2.set_ylabel("Dice")
    ax2.grid(axis="y", alpha=0.3)
    ax2.set_axisbelow(True)
    ax2.legend(
        handles=[
            Patch(facecolor=str(NOISE_LIGHTNESS[noise]), label=noise)
            for noise in NOISE_ORDER
        ],
        title="Noise (light to dark)",
    )

    # ------------------------------------------------------------------------
    # Save / display
    # ------------------------------------------------------------------------

    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Plot gespeichert: {save_path}")

    plt.show()


# ============================================================================
# Main
# ============================================================================

def main():
    # ------------------------------------------------------------------------
    # Arguments
    # ------------------------------------------------------------------------

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", help="W&B CSV export path")
    parser.add_argument("--epochs-per-iteration", type=int, default=50)
    parser.add_argument("--save", help="Optional output PNG/PDF/SVG filename")

    args = parser.parse_args()

    if args.epochs_per_iteration <= 0:
        parser.error("--epochs-per-iteration muss positiv sein")

    # ------------------------------------------------------------------------
    # Read metrics / print summary
    # ------------------------------------------------------------------------

    results = read_grouped_metrics(args.csv, args.epochs_per_iteration)

    print("\nGruppe             Seeds  Iterationen  Best mean Dice")
    print("-" * 57)

    for (ablation_id, noise), result in results.items():
        print(
            f"{ablation_id + ' / ' + noise:<19} {len(result['seeds']):>2}"
            f" {len(result['per_iteration']):>11}  {result['best_overall']:.6f}"
        )

    # ------------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------------

    plot_results(results, args.save)


if __name__ == "__main__":
    main()
