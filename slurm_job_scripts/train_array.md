# Training array

Submit from the repository root using a config path relative to `conf/`:

```bash
mkdir -p logs
sbatch slurm_job_scripts/train_array.sbatch ablation/pre_a/pre_a57
```

- Starts two independent runs with seeds **42, 1042**.
- Each task stages the data and runs all configured training iterations.
- Each task requests one A100 and 16 CPUs; up to two tasks run concurrently.
- Logs: `logs/<job-name>_<job-id>.out` and `.err`.

Include `${train.seed}` and `${oc.env:SLURM_JOB_ID,local}` in `experiment.name` for unique run names. For fresh Noisy Student runs, use `start_iteration: 0`, `resume_output_dir: auto`, and `resume_run: auto` under `train`.
