# Ablation plots

`plot_ablation_dice.py` compares validation Dice scores from a W&B CSV export by ablation ID and noise level. It averages seeds per epoch and plots the best mean Dice per NST iteration (50 epochs by default) and overall. Incomplete iterations are skipped.

Requires `pandas`, `numpy`, and `matplotlib`.

```bash
python utils/ablation/plot_ablation_dice.py wandb_export.csv --save dice.png
```
