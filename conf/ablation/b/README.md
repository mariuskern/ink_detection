# Ablation B

Each ID has `low`, `medium`, and `high` variants, e.g. `b4_medium.yaml`.
Each concrete config imports the local `noisy_student.yaml` and defines its
own checkpoint selection and aggregation under `train.teacher`. As in A, local
`base.yaml` inherits `../common/base.yaml`; local `noisy_student.yaml` inherits
the adjusted local base and `../common/noisy_student.yaml`. There are no
intermediate `b0.yaml`–`b4.yaml` configs.
Model and input-noise presets, data splits, shared training defaults, and shared
Noisy Student settings live in `../common/` and are reused by A and B. B imports
the common presets directly, including the growing UNETR++ model schedule:
S → M → L → L → L → L. B-specific paths, W&B settings, checkpoint counts, and
teacher selection stay local. The existing A0 baseline checkpoints are reused
for iteration 0, matching A4.

| ID | Used checkpoints | Aggregation |
| --- | --- | --- |
| b0 | Best k (validation Dice) | Weight average |
| b1 | Best k (validation Dice) | Individual teachers |
| b2 | Last k (SWA window ending at the best checkpoint) | Weight average |
| b3 | Last k (SWA window ending at the best checkpoint) | Individual teachers |
| b4 | Best k + Last k | One weight average per group; two teachers |

B5 is intentionally omitted. “Last k” means k checkpoints in total: the best
checkpoint plus the k−1 immediately preceding checkpoint epochs, selected by the
existing SWA catalog. For k = 5 and best epoch e, this is e−4 through e, inclusive.
If fewer preceding checkpoints exist, all available checkpoints up to and
including the best are used.
B4 forms two separate weight averages: one over Best-k and one over Last-k.
The resulting two teachers' predictions are aggregated by the existing pipeline.
There is no single weight average over both groups.

## Settings

- `ablation.k: 5` controls both `save_top_k` and `swa_save_k`.
  Both catalogs are retained for every variant; teacher flags select the groups.
  SWA can contain fewer than k checkpoints when the best epoch occurs early.
- Each run starts at iteration 1 using the A0 baseline matching its noise and
  seed, followed by five student iterations (1–5). The teacher flags select
  Best-k and/or SWA-k from that baseline's retained catalogs as well.
- Each `b0`–`b4` noise variant explicitly sets `ablation.baseline_id: a0` in
  its own config; the shared Noisy Student config only defines `teacher_root`.
- The model schedule restores the S teacher at iteration 0, trains M at
  iteration 1, and uses L for iterations 2–5, matching A4.
- A4 settings: 50 epochs per iteration, batch size 15, prediction batch size 80,
  and the same 3 labeled, 7 unlabeled, and 3 validation segments.
- Low, medium, and high input/model noise match the corresponding A4 variants.
- Refinement settings match A4: mask refinement is enabled for real labels
  and validation masks with margin factor 0.25 and closing factor 0.4. Label
  refinement is intentionally disabled (`train.label_refinement.enabled: false`).
  Its inactive parameters remain radius factor 0.08, minimum value 0.5, and
  background smoothing enabled.
- Pseudo-label refinement uses only threshold 0.5, confidence 0.8, and uncertainty
  0.15. No additional postprocessing, such as removing small components, is used.
- TTA is disabled (`tta_num: 1`, `aug.variants: []`).
- W&B project: `ink_ablation_b`. Output and temporary paths use `ablation_b`,
  the experiment ID, noise, seed, timestamp, and Slurm IDs to isolate runs.

## Run

From the repository root, with the existing cluster environment and `.env`:

Reuse the A0 baselines for all noise levels and seeds, retaining at least k
checkpoints per selected catalog unless the documented early-best case applies.
A completed A4 Noisy Student run is not required; B reuses A4's configuration
and trains its own student iterations. The default
teacher checkpoint directory is:

```text
${OUTPUT_DIR}/ablation_a/supervised/a0_${ablation.noise}/seed_${train.seed}/checkpoints
```

Override `ablation.teacher_root` if the baseline folders are stored elsewhere,
or `train.teacher_checkpoint` to use a specific checkpoint source.

```bash
mkdir -p logs
for id in b0 b1 b2 b3 b4; do
    for noise in low medium high; do
        sbatch --job-name="${id}_${noise}" slurm_job_scripts/train_array.sbatch "ablation/b/${id}_${noise}"
    done
done
```

Each array runs seeds **42, 1042, 2042, 3042, 4042**: 15 configurations and
75 independent runs. Each task requests one A100, 16 CPUs, and 23:59 hours.

Preview B4 without training:

```bash
python train.py --config-name ablation/b/b4_medium --cfg job --resolve
```

To change k for a local run, append `ablation.k=3`, for example. For resume at
an iteration boundary, retain the config and seed and set
`train.start_iteration=<next_iteration>`, `train.teacher_checkpoint=auto`,
`train.resume_output_dir=/path/to/existing/b_run`, and `train.resume_run=auto`.
Each array seed requires its own resume folder.
