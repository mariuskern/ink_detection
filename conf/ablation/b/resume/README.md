# Ablation B continuation

All 75 configs now use `train.iterations: 7`: the common final student
iteration is **6** (baseline iteration 0, student iterations 1–6).
This replaces the previous final iteration 7 (`iterations: 8`).

The saved runs were checked on 2026-11-16. Each config keeps its existing
seed, output directory, W&B run, teacher selection, and restart boundary.
All 50 epochs are retained for every student iteration.

| Variants | Runs | Last completed iteration | Restart at | Iterations to train | Saved teachers |
| --- | ---: | ---: | ---: | --- | --- |
| B0, B2 | 30 | 5 | 6 | 6 | One weight average |
| B1, B3 | 30 | 4 | 5 | 5, 6 | Individual checkpoints, up to five |
| B4 | 15 | 5 | 6 | 6 | Two group weight averages |

Incomplete iteration 5 in B1/B3 is restarted from its beginning; partial
epochs do not count as a completed student iteration. Existing iteration
directories are backed up by the training code.

## Time budget

Estimates assume the existing `train.sbatch` allocation: one A100 and
23:59 hours. These are estimates for a fresh continuation, not measured
W&B resume times for B.

| Variants | Estimated W&B startup | Training + exports per new iteration | Estimated total job time |
| --- | --- | --- | --- |
| B0, B2 | 10.7 h | 3.4–3.5 h | 15.4–15.5 h |
| B1, B3 | 8.5 h | 5.3–5.4 h | 22.2–22.7 h |
| B4 | 10.7 h | 4.0 h | 16.4–16.6 h |

Training and export estimates come from each run's retained checkpoint and
export timestamps. Exports include predictions and pseudo labels, using
the actual individual or averaged teacher selection. Totals also allow for
initial pseudo-label regeneration, training compilation, and 30 minutes
for staging and other startup work. For initial regeneration, the full
measured export duration is used as a conservative proxy.

The W&B estimate scales the slower measured A4 continuation startup
(11.74 h for approximately 4.12 million history rows) to B's completed
history: approximately 2.99 million rows before iteration 5 or 3.74 million
before iteration 6. B uses fewer labeled/unlabeled/prediction segments
(3/7/3 instead of 5/17/5), but still uses 4,000 training samples per epoch
and the same three validation segments. Most history rows are per-batch
metrics, so startup is **not** scaled by the unlabeled-segment ratio 7/17.
The smaller number of completed iterations is included in the row counts.

The stress estimate adds 15% to W&B startup and 5% to each new iteration.
It reaches about 24.4 h for B1/B3, so completion within one allocation is
not guaranteed. A common final iteration 7 would already require about
27.5–28.1 h for B1/B3 before those extra margins. Iteration 6 is the smallest
common final iteration that continues every run.

## Submit

Run from the repository root with the existing environment and `.env`.
Use the single-job script because each config already fixes its seed:

```bash
for config in conf/ablation/b/resume/*.yaml; do
    name=${config##*/}
    sbatch --job-name="${name%.yaml}_resume" slurm_job_scripts/train.sbatch \
        "ablation/b/resume/${name%.yaml}"
done
```

If a B1/B3 continuation reaches the time limit after completing iteration 5
and saving its teacher selection, resubmit its same config with
`train.start_iteration=6` passed to `train.py`. Check that
`iteration/5/checkpoints/teacher_checkpoints.json` and every listed checkpoint
exist first. Keep `train.iterations=7` so its final target stays equal to the
other runs. The current `train.sbatch` accepts only the config name; it does
not forward additional training overrides. Use a job script that passes the
override or update that run's resume config after checking its new progress.
