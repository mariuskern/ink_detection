# Pre-ablation configurations

The preliminary experiment series live here, separately from `conf/ablation/`:

- `pre_a/`: the original Pre-A experiments and their local presets.
- `pre_b/`: the original Pre-B experiments and their local presets.
- `pre_c/`: the original Pre-C experiments and their local presets.

Use config names relative to `conf/`, without the `.yaml` extension:

```bash
python train.py --config-name pre_ablations/pre_b/pre_b1 --cfg job --resolve
sbatch slurm_job_scripts/train_array.sbatch pre_ablations/pre_b/pre_b1
```

Only the configuration locations and absolute model-schedule paths changed.
Training settings, experiment names, W&B projects, output paths, and temporary
paths retain their previous values. Numbered submission loops are documented in
`../../slurm_job_scripts/README.md`.
