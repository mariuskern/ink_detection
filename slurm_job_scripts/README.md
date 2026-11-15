# Submitting Slurm jobs

Run the following Bash commands on the cluster from the repository root.
Create the log directory before submitting jobs:

```bash
mkdir -p logs
```

## Single training job

Pass a config path relative to `conf/`, without the `.yaml` extension:

```bash
sbatch slurm_job_scripts/train.sbatch ablation/a/a1_medium
```

Without a config argument, `train.sbatch` uses `config`. Each job stages the
data and then starts training. It requests one A100, 16 CPUs, and up to
23 hours 59 minutes.

## Custom job names

Use `--job-name` (or `-J`) before the script path to override the script's
default Slurm job name:

```bash
sbatch --job-name=a1_medium slurm_job_scripts/train.sbatch ablation/a/a1_medium
```

The name appears in the queue and log filenames:
`logs/<job-name>_<job-id>.out` and `.err`. It does not change the Hydra
config or `experiment.name`.

## Multiple jobs in a loop

Submit a consecutive range of supervised baseline configs. Bash includes
both endpoints: `{0..2}` submits A0, A1, and A2 at medium noise.

```bash
for i in {0..2}; do
    config="a${i}_medium"
    sbatch --job-name="$config" slurm_job_scripts/train.sbatch "ablation/a/$config"
done
```

Or submit an explicit list:

```bash
for config in a0_medium a1_medium a2_medium; do
    sbatch --job-name="$config" slurm_job_scripts/train.sbatch "ablation/a/$config"
done
```

Each `sbatch` call submits a separate job. These loops do not limit how many
jobs can run concurrently; Slurm schedules them according to available
resources and account limits.

## Five training runs as an array

```bash
sbatch --job-name=a1_medium_seeds slurm_job_scripts/train_array.sbatch ablation/a/a1_medium
```

The array submits five independent runs of the same config, with seeds
**42, 1042, 2042, 3042, 4042**. `--array=0-4%5` allows up to five tasks to
run concurrently, subject to resource availability. Each task requests one
A100 and 16 CPUs, stages the data, and runs all configured training iterations.
Only `train.seed` is overridden.

To submit five runs for each of several configs:

```bash
for config in a0_medium a1_medium a2_medium; do
    sbatch --job-name="${config}_seeds" slurm_job_scripts/train_array.sbatch "ablation/a/$config"
done
```

The concurrency limit applies separately to each array; this example submits
15 tasks in total, with up to 15 able to run concurrently.

Include `${train.seed}` and `${oc.env:SLURM_JOB_ID,local}` in
`experiment.name` for unique run names. For fresh Noisy Student runs, use
`start_iteration: 0`, `resume_output_dir: auto`, and `resume_run: auto` under
`train`. Concurrent experiments that write the same pseudo-label versions
must use isolated data storage; see the
[training reference](../conf/README.md#noisy-student-training).

## Submit numbered experiments, skipping missing configs

Pre-ablation configs live under `conf/pre_ablations/`, outside `conf/ablation/`.
Check run numbers 1 through 60 and skip missing YAML files. Change `series`
and the range to reuse the loop for another pre-ablation.
Job names follow the config names (`pre_b1`, `pre_b5`, etc.):

```bash
mkdir -p logs
series="pre_b"
for i in {1..60}; do
    config="${series}${i}"
    config_path="pre_ablations/${series}/${config}"
    [[ -f "conf/${config_path}.yaml" ]] || continue
    sbatch --job-name="$config" slurm_job_scripts/train.sbatch "$config_path"
done
```

For five seeds per experiment, use the `_seeds` job-name convention:

```bash
mkdir -p logs
series="pre_b"
for i in {1..60}; do
    config="${series}${i}"
    config_path="pre_ablations/${series}/${config}"
    [[ -f "conf/${config_path}.yaml" ]] || continue
    sbatch --job-name="${config}_seeds" slurm_job_scripts/train_array.sbatch "$config_path"
done
```

This submits one array per existing config, with five seeds per array.
The concurrency limit applies separately to each array.

## Other scripts

- `train_a40.sbatch` requests one A40 and currently uses the fixed `config`
  config; it does not accept a config argument.
- `preprocess.sbatch` runs data preparation with cluster-specific paths.
  Create its log directory with `mkdir -p logs/preprocess` before submitting
  it with `sbatch slurm_job_scripts/preprocess.sbatch`.

## Check or cancel jobs

```bash
squeue -u "$USER"
scancel JOB_ID
```

Replace `JOB_ID` with the job number to cancel.
For an array, cancelling its array job ID cancels all its tasks.
