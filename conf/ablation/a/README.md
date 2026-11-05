# Ablation a

Each ID has `low`, `medium`, and `high` variants, e.g. `a3_medium.yaml`.
Every Slurm array runs five seeds: **42, 1042, 2042, 3042, 4042**.

| ID | Experiment | Model sizes, iterations 0–5 |
| --- | --- | --- |
| a0 | Supervised S | S |
| a1 | Supervised M | M |
| a2 | Supervised L | L |
| a3 | Noisy Student; a2 baseline | L → L → L → L → L → L |
| a4 | Noisy Student; a0 baseline | S → M → L → L → L → L |

The six iterations comprise the supervised baseline at iteration 0 and five student
iterations (1–5). Noisy Student starts at iteration 1 using the matching baseline's size, noise,
and seed. There are 45 baseline runs and 30 Noisy Student runs.

## Defaults

- Frozen pre_a–pre_c settings and data splits: 3 labeled, 7 unlabeled, 3 validation segments.
- 50 epochs per training; batch size 15 for training/validation, 80 for predictions.
- Validation Dice selects checkpoints; `deterministic: false`.
- Baselines retain Top-10 and SWA-10; this ablation uses Top-1 teachers.
- All supervised and Noisy Student runs share the W&B project `ink_ablation_a`.
- Mask refinement: margin factor 0.25, closing factor 0.4; label refinement disabled.
- Standard pseudo-label refinement: threshold 0.5, confidence 0.8, uncertainty 0.15.
- Noise pairs: low/reduced model noise, medium/default model noise, high/model_noise_more.

## Run

From the repository root, with the existing cluster environment and `.env`:

```bash
mkdir -p logs
for id in a0 a1 a2; do
    for noise in low medium high; do
        sbatch --job-name="${id}_${noise}" slurm_job_scripts/train_array.sbatch "ablation/a/${id}_${noise}"
    done
done
```

After the baselines finish, repeat the loop with `a3 a4` instead of `a0 a1 a2`.
Each task requests one A100, 16 CPUs, and 23:59 hours.

Baseline checkpoints use fixed paths, for example:

```text
${OUTPUT_DIR}/ablation_a/supervised/a0_low/seed_42/checkpoints/
```

`ablation.teacher_root` already points to `${OUTPUT_DIR}/ablation_a/supervised`.
Teacher paths automatically include the matching baseline ID, noise, and seed.
Keep baseline paths stable so Noisy Student can find the matching teachers without
overrides. If a baseline is stored elsewhere, override `ablation.teacher_root` or
`train.teacher_checkpoint` accordingly.

Noisy Student adds a timestamp, job ID, and array task ID to each run's path:

```text
${OUTPUT_DIR}/ablation_a/noisy_student/a4_medium/seed_42_time_<timestamp>_job_id_<job_id>_task_<task_id>/iteration/<i>/checkpoints/
```

Baselines use one stable output folder per ablation ID, noise level, and seed;
new Noisy Student runs get separate folders. Local runs use `local` for the job ID
and `none` for the array task ID. `experiment.name` includes a timestamp and Slurm
identifiers for W&B, but output paths are configured independently of that name.
A fresh run rejects an occupied output folder. To repeat a baseline as a separate
experiment, override `path.output_dir` with a new destination and point its
consumers to the new teacher location.

Noisy Student temporary files use the same run-specific suffix under `${TMP_DIR}`:
`${TMP_DIR}/ablation_a/noisy_student/${ablation.id}_${ablation.noise}/seed_${train.seed}_time_<timestamp>_job_id_<job_id>_task_<task_id>`.
Separate runs therefore also have separate temporary folders. Resume may use a
fresh temporary folder: pseudo-labels are regenerated from the saved teachers
before student training continues.

## Reuse checkpoints

Supply **one folder** to select teachers through the config, for example:

```yaml
train:
  checkpoint:
    save_top_k: 3
    swa_save_k: 5
  teacher:
    use_top_k: true
    top_k_mode: individual
    use_swa: true
    swa_mode: average
```

Both groups support `individual` and `average`; weight averages are computed
separately. SWA includes the best checkpoint and up to k−1 predecessors, using
fewer if the best epoch is early. Top-k similarly uses fewer checkpoints when
training ended before k checkpoint epochs. `checkpoint_potential_maximum_k.json`
records the eligible checkpoint counts at training end (Top-k) and at the best
epoch (SWA), accounting for the checkpoint interval. A short catalog is accepted
only when these counts confirm the early case. Missing selected files or an
insufficiently retained set raise an error. Save a larger baseline set beforehand
if needed. Older catalogs without availability metadata must contain at least k
checkpoints, or be supplied as explicit checkpoint files.
`save_top_k: -1` requires availability metadata confirming a complete Top-k
catalog; truncated or unverified catalogs are rejected.

`top_k_checkpoints.json` and `swa_checkpoints.json` catalog the retained files.
`teacher_checkpoints.json` records the actual teacher selection for exact resume.
New experiments select from the catalogs regardless of any existing teacher selection.
Files are copied with their original names; the source run stays unchanged.

Alternatively, supply one checkpoint file or a list of `.pt`/`.pth`/`.ckpt` files.
Different checkpoint files must have distinct resolved filenames; repeated paths
to the same checkpoint are allowed.
These are used exactly as given for the initial pseudo-label generation.
Later iterations always select their own teachers through the config.
Folder lists, mixed folder/file lists, and explicit JSON inputs are rejected.

## Resume in the same run

To continue after iteration 2, keep the original config and seed and set:

```yaml
train:
  start_iteration: 3
  teacher_checkpoint: auto
  resume_output_dir: /path/to/existing/noisy_student_run
  resume_run: auto
```

Existing iteration folders are backed up; the exact ensemble from
`teacher_checkpoints.json`, saved weight averages and the W&B run are reused.
The teacher manifest and its selected files must exist; resume does not rebuild
the selection from catalogs. Raw checkpoints and catalogs are preserved too.
Resume restarts at an iteration boundary. Each array seed needs its own resume
folder. `path.output_dir` automatically uses `train.resume_output_dir` when an
existing run is specified; Hydra logs and checkpoints go into that same folder.
`resume_run: auto` restores the W&B identity saved in that folder; it does not
specify a filesystem path. Temporary files may use a fresh folder.
For a new Noisy Student experiment, keep `resume_output_dir: auto`; the default
output path creates a separate folder for each run. No explicit
`path.output_dir` override is needed for resume.

Preview without training:

```bash
python train.py --config-name ablation/a/a4_medium --cfg job --resolve
```
