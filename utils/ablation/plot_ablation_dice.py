#!/usr/bin/env python3
"""Plot W&B validation Dice by ablation ID and noise, averaging seeds first.

Usage:
    python utils/ablation/plot_ablation_dice.py wandb_export.csv
    python utils/ablation/plot_ablation_dice.py wandb_export.csv --show-counts --save dice.png
    python utils/ablation/plot_ablation_dice.py wandb_export.csv --show-values --max-iteration 3

Requires: pandas, numpy, matplotlib

Each iteration averages only runs with values for every expected epoch.
Iterations without complete runs carry forward the last available iteration's
value and are marked with an x. Leading missing iterations remain empty.
With --show-counts, each point is labeled with the number of complete runs in
that iteration. Carried points have n=0.
With --show-values, each point is labeled with its Dice value (six decimals).
With --max-iteration N, only iterations 1 through N contribute to the results,
including the overall best values.
"""

import argparse
import colorsys
import re
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from matplotlib.lines import Line2D


# ============================================================================
# Constants
# ============================================================================

# Example: b4_high_seed_4042_20261113_... - metrics_val/dice
DICE_COLUMN_RE = re.compile(
    r"^(?P<id>[ab]\d+)_(?P<noise>low|medium|high)_seed_(?P<seed>\d+)_.* - metrics_val/dice$"
)
NOISE_ORDER = ("low", "medium", "high")
NOISE_LIGHTNESS = {"low": 0.75, "medium": 0.50, "high": 0.30}


# ============================================================================
# Read and aggregate metrics
# ============================================================================

def read_grouped_metrics(csv_path, epochs_per_iteration, max_iteration=None):
    # ------------------------------------------------------------------------
    # CSV input / validation
    # ------------------------------------------------------------------------

    if max_iteration is not None and max_iteration <= 0:
        raise ValueError("max_iteration muss positiv sein.")

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

    if max_iteration is not None:
        df = df.loc[df.index < max_iteration * epochs_per_iteration]
        if not (df.index >= 0).any():
            raise ValueError(
                f"Keine Epochen bis einschließlich Iteration {max_iteration} gefunden."
            )

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

        iteration_max = {}
        iteration_counts = {}
        iterations = sorted(set(int(step // epochs_per_iteration) for step in run_df.index if step >= 0))

        for idx in iterations:
            expected_steps = np.arange(
                idx * epochs_per_iteration, (idx + 1) * epochs_per_iteration
            )
            block = run_df.reindex(expected_steps)
            complete_runs = block.notna().all(axis=0)
            count = int(complete_runs.sum())
            iteration_counts[idx + 1] = count

            if count < len(seed_series):
                print(
                    f"WARNUNG: {key[0]} / {key[1]}: Iteration {idx + 1} "
                    f"hat {count}/{len(seed_series)} vollständige Runs."
                )

            if count == 0:
                continue

            # Use the same complete runs at every epoch of this iteration.
            mean_dice = block.loc[:, complete_runs].mean(axis=1)
            iteration_max[idx + 1] = float(mean_dice.max())

        if not iteration_max:
            print(f"WARNUNG: Keine vollständige Iteration für {key[0]} / {key[1]}.")
            continue

        results[key] = {
            "per_iteration": iteration_max,
            "per_iteration_counts": iteration_counts,
            "best_overall": max(iteration_max.values()),
            "seeds": sorted(seed_series),
        }

    if not results:
        raise ValueError(
            "Keine vollständigen Iterationen gefunden. Prüfe, ob der CSV-Export "
            "alle 50 Validation-Epochen je Iteration enthält."
        )

    # Use the shared CSV horizon, including iterations absent from shorter runs.
    last_iteration = max(int(step // epochs_per_iteration) + 1 for step in df.index if step >= 0)
    for result in results.values():
        per_iteration = result["per_iteration"]
        carried_iterations = []
        previous = None
        for iteration in range(1, last_iteration + 1):
            result["per_iteration_counts"].setdefault(iteration, 0)
            if iteration in per_iteration:
                previous = per_iteration[iteration]
            elif previous is not None:
                per_iteration[iteration] = previous
                carried_iterations.append(iteration)
        result["per_iteration"] = dict(sorted(per_iteration.items()))
        result["carried_iterations"] = carried_iterations

    return results


# ============================================================================
# Plot results
# ============================================================================

def plot_results(results, save_path=None, show_counts=False, show_values=False):
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
        carried = set(result.get("carried_iterations", []))

        ax1.plot(
            list(per_iteration),
            list(per_iteration.values()),
            label=f"{ablation_id} / {noise} (total seeds={len(result['seeds'])})",
            color=colors[(ablation_id, noise)],
            linestyle=line_styles.get(ablation_id, "-"),
            marker="o",
            markevery=[index for index, iteration in enumerate(per_iteration) if iteration not in carried],
            markersize=4,
        )
        if carried:
            ax1.plot(
                sorted(carried),
                [per_iteration[iteration] for iteration in sorted(carried)],
                color=colors[(ablation_id, noise)],
                linestyle="none",
                marker="x",
                markersize=7,
                markeredgewidth=1.5,
            )

        if show_counts or show_values:
            for iteration, dice in per_iteration.items():
                labels = []
                if show_values:
                    labels.append(f"{dice:.6f}")
                if show_counts:
                    labels.append(f"n={result['per_iteration_counts'][iteration]}")
                ax1.annotate(
                    "\n".join(labels),
                    xy=(iteration, dice),
                    xytext=(5, 5),
                    textcoords="offset points",
                    fontsize=8,
                    color=colors[(ablation_id, noise)],
                )

    ax1.set_title("Best mean Dice per NST iteration")
    ax1.set_xlabel("NST iteration")
    ax1.set_ylabel("Dice")
    ax1.set_xticks(sorted({i for r in results.values() for i in r["per_iteration"]}))
    ax1.grid(alpha=0.3)
    handles, labels = ax1.get_legend_handles_labels()
    if any(r.get("carried_iterations") for r in results.values()):
        handles.append(Line2D([], [], color="black", marker="x", linestyle="none", markersize=7))
        labels.append("Carried forward (missing/incomplete iteration)")
    ax1.legend(handles, labels, fontsize=8, loc="best", ncol=2)

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
    parser.add_argument(
        "--show-counts",
        action="store_true",
        help="Show the number of complete runs in each iteration next to each point",
    )
    parser.add_argument(
        "--show-values",
        action="store_true",
        help="Show the Dice value (six decimals) next to each point",
    )
    parser.add_argument(
        "--max-iteration",
        type=int,
        help="Include only NST iterations up to this number (inclusive, starting at 1)",
    )

    args = parser.parse_args()

    if args.epochs_per_iteration <= 0:
        parser.error("--epochs-per-iteration muss positiv sein")
    if args.max_iteration is not None and args.max_iteration <= 0:
        parser.error("--max-iteration muss positiv sein")

    # ------------------------------------------------------------------------
    # Read metrics / print summary
    # ------------------------------------------------------------------------

    results = read_grouped_metrics(
        args.csv, args.epochs_per_iteration, max_iteration=args.max_iteration
    )

    print("\nGruppe             Seeds  Iterationen  Best mean Dice")
    print("-" * 57)

    for (ablation_id, noise), result in results.items():
        print(
            f"{ablation_id + ' / ' + noise:<19} {len(result['seeds']):>2}"
            f" {len(result['per_iteration']) - len(result['carried_iterations']):>11}  {result['best_overall']:.6f}"
            f"  (+{len(result['carried_iterations'])} fortgeschrieben)"
        )

    # ------------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------------

    plot_results(
        results, args.save, show_counts=args.show_counts, show_values=args.show_values
    )


if __name__ == "__main__":
    main()
