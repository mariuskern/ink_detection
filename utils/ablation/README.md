# Ablation plots

`plot_ablation_dice.py` compares validation Dice scores from a W&B CSV export by ablation ID and noise level. For each NST iteration (50 epochs by default), it selects runs with values for every expected epoch, averages those seeds per epoch, and plots the best mean Dice for that iteration and overall. A run with even one missing value is excluded from the entire iteration.

Requires `pandas`, `numpy`, and `matplotlib`.

```bash
python utils/ablation/plot_ablation_dice.py wandb_export.csv --save dice.png
```

Add `--show-counts` to label each point in the iteration plot with `n=<count>`:

```bash
python utils/ablation/plot_ablation_dice.py wandb_export.csv --show-counts --save dice.png
```

The count is the number of runs that completed every expected epoch in that
iteration, so it can vary between iterations. If no run is complete, the point
carries forward the last available iteration's value, is marked with an `x`, and
shows `n=0`. Leading iterations without complete runs remain empty.
