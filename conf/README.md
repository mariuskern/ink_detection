# Configuration and training reference

For a project overview and installation, see the [main README](../README.md).
Input formats and conversion are covered in the [data guide](../data/README.md).
Run the commands below from the repository root.

## How configuration works

The project uses Hydra to compose YAML files from this directory. Select an
entry config with `--config-name`, without the `.yaml` extension. Select a
configuration group with `group=preset`, and override individual fields with
`section.field=value`:

```bash
python train.py --config-name=config path=local train=supervised train.max_epochs=10
```

Preview the full configuration without starting training:

```bash
python train.py --config-name=config path=local --cfg job --resolve
```

This still imports the Python dependencies and reads environment variables;
activate the environment and configure `.env` first.

| Entry config | Purpose |
| --- | --- |
| `config` / `config_default` | UNETR++ with Noisy Student training by default. |
| `config_unet` | Vesuvius U-Net with its dataset and training presets. |
| `config_resnet3d` | Vesuvius ResNet3D with its dataset, loss, and training presets. |
| `debug`, `debug_unet`, `debug_resnet3d` | Development profiles; inspect their overrides before use. |
| `predict` | Standalone prediction via `predict.py`. |
| `ablation/a/*`, `ablation/b/*` | Controlled experiment profiles with their own paths and teacher settings. |

## Configuration groups

| Directory | Controls |
| --- | --- |
| `model/` | Architecture, model capacity, dropout, and depth projection. |
| `data/dataset/` | Tile size and stride, input depth, center slice, normalization, and tile eligibility. |
| `data/sampler/` | Sampling of real and pseudo-labeled tiles. |
| `data/dataloader/` | Batch sizes, worker count, and prefetching. |
| `train/` | Data splits, optimizer settings, precision, checkpoint retention, and training mode. |
| `aug/` | Student/supervised augmentations and teacher test-time augmentation settings. |
| `refinement/` | Pseudo-label thresholding and confidence/uncertainty filtering. |
| `loss/` | Loss functions and their parameters. |
| `path/` | Data staging, data location, output folder, and temporary storage. |
| `wandb/` | Weights & Biases project settings. |

Dataset, sampler, and dataloader groups are mounted at the top level as
`dataset`, `sampler`, and `dataloader`. For example, override
`dataset.tile_size=192` or `dataloader.batch_size=4`.
To select another dataset preset, use `data/dataset@dataset=<preset>`.

`train.labeled` supplies manual training labels; `train.unlabeled` supplies
segments for pseudo-label training; `train.val` defines validation segments.
`train.predict` selects additional prediction exports during Noisy Student
training. Standalone prediction uses `train.segments` instead.
Segment IDs include the scroll prefix, such as `0172_20241025145341`.

Model and dataset presets must be compatible. The `config_unet` and
`config_resnet3d` entry configs select their associated settings together.
In Noisy Student training, `train.model_schedule` can select model presets by
iteration; its final entry is reused for later iterations. Explicit
`model.<field>` CLI overrides are applied to each scheduled model.

## Paths and environment

Create `.env` at the repository root with absolute paths:

```dotenv
DATA_DIR=/absolute/path/to/data
CODE_DIR=/absolute/path/to/ink_detection
OUTPUT_DIR=/absolute/path/to/outputs
TMP_DIR=/absolute/path/to/scratch
```

| Profile | Data source | Intended use |
| --- | --- | --- |
| `path=local` | `${DATA_DIR}/data_zarr`; no staging | Read prepared data directly. |
| `path=alex` (training default) | Stage `${DATA_DIR}/data_tar` into `${TMPDIR}/data_zarr` | Cluster jobs with node-local scratch. |
| `path=predict` (prediction default) | `${DATA_DIR}/data_zarr`; no staging | Standalone inference. |

`TMPDIR` is the cluster-provided variable used by `path=alex`; `TMP_DIR` is
the project variable for temporary run files. They have different roles.
`CODE_DIR` is used by helper scripts; the Python entry points load `.env`.

For the standard local training profiles, new runs write to
`${OUTPUT_DIR}/${train.name}/${model.name}/${experiment.name}`.
Each run records `.hydra/config.yaml` and `config.resolved.yaml`.
Temporary run files go beneath `${TMP_DIR}`. Ablation profiles define their
own output and temporary paths; consult their READMEs.

Training logs through W&B. Use `wandb login` for online logging, or set
`WANDB_MODE=offline` for a fresh offline experiment. Resuming an existing run
requires access to its synced W&B history, as described below.

## Supervised training

For cluster submission, job names, loops, and seed arrays, see
[Submitting Slurm jobs](../slurm_job_scripts/README.md).

Run `python train.py --config-name=<config_name> train=supervised`.
Training uses `train.labeled`; validation uses `train.val`.

To include fixed pseudo labels from `train.unlabeled`, set both
`train.pseudo_label_version` and `train.pseudo_data_dir`; choose hard or soft
targets with `train.pseudo_target`. With both source fields set to null,
`unlabeled` is ignored. Pseudo labels remain fixed throughout the run.

For pseudo-only training, use `train.labeled=[]`, `sampler.real_fraction=null`
and `sampler.limit_pseudo_tiles=false`.

## Noisy Student training

Set these options in [`train/noisy_student.yaml`](train/noisy_student.yaml) or an inherited variant.
For CLI overrides, prefix them with `train.`, e.g. `train.start_iteration=2`.
Run `python train.py --config-name=<config_name>` with your training profile.

| Goal | Set explicitly | Leave on `auto` |
| --- | --- | --- |
| Start from scratch | `start_iteration: 0` (default) | `teacher_checkpoint`, `resume_output_dir`, `resume_run` |
| Resume an existing experiment | `resume_output_dir` and `start_iteration` | `teacher_checkpoint`, `resume_run` |
| Start a new experiment from teachers | `teacher_checkpoint` and `start_iteration` | `resume_output_dir`, `resume_run` |

### Options

| Option | Default | Meaning |
| --- | --- | --- |
| `start_iteration` | `0` | First iteration to train from the beginning. For `N > 0`, supply a teacher from `N - 1`. |
| `iterations` | `5` | Exclusive upper bound: start `2`, end `5` trains `2`, `3`, `4`. |
| `teacher_checkpoint` | `auto` | Read teachers from an explicit `resume_output_dir`; alternatively supply a file, list, or teacher folder. |
| `resume_output_dir` | `auto` | Select the output folder; see below. |
| `resume_run` | `auto` | Select the W&B run; see below. |

| Option | Value | Effect |
| --- | --- | --- |
| `teacher_checkpoint` | `auto` | At iteration `0`: no teacher. At `N > 0`: select teachers from iteration `N - 1` in an explicitly supplied `resume_output_dir`. |
| | `null` | No explicit teacher; at `N > 0`, also falls back to an explicit `resume_output_dir`. |
| | `"path/model.pt"` | Use this checkpoint to generate pseudo-labels. |
| | `["path/a.pt", "path/b.pt"]` | Use these checkpoints as the teacher ensemble. |
| | `"outputs/source_run"` | Select teachers from iteration `N - 1`; an iteration/checkpoint folder can also be supplied directly. |
| `resume_output_dir` | `auto` | Create a new folder based on `experiment.name`, including when checkpoints are supplied. |
| | `null` | Same as `auto`; accepted for compatibility. Prefer `auto` in new configs. |
| | `checkpoint` | Infer the run folder from the teacher; requires `<run>/iteration/<number>/checkpoints/<file>`. |
| | `"outputs/old_run"` | Resume in this folder; automatic teachers come from its iteration `N - 1`, and automatic W&B selection uses its saved run. |
| `resume_run` | `auto` | Resume the saved W&B run for an explicit output folder (or `checkpoint`); `auto`/`null` output creates a new run. |
| | `null` | New experiments only; rejected for continuations. |
| | `"entity/project/run_id"` | Continuations only: specify the same saved run, or supply the identity for a legacy run. |

**How `auto` works:**

1. `resume_output_dir: auto` chooses the regular folder using `experiment.name`;
   it does not infer an old folder from supplied checkpoints. `null` has exactly
   the same behavior and is retained as a compatibility alias.
2. `teacher_checkpoint: auto` uses no teacher at iteration `0`. For `N > 0`,
   provide an existing run as `resume_output_dir`, or supply explicit teachers.
   If neither is available, startup fails.
3. `resume_run: auto` creates a new W&B run when `resume_output_dir` is `auto` or
   `null`. For an existing run folder, it reads the saved W&B identity there.
   Missing identities require an explicit run ID; they are not guessed.

An explicit teacher overrides automatic selection. A continuation must reuse
both the old output folder and its W&B run; logging always appends. New output
folders always start new W&B runs. Contradictory combinations raise an error. Hydra and training share the output
folder; `path.tmp_dir` is independent. Use a unique `experiment.name` or its timestamp default for a
new experiment.

### Examples

Each example lists **all five options** above. Other training/model settings
are inherited from the selected profile.

#### 1. Start a normal run from iteration 0

Train iterations `0` through `4` in a new output folder and W&B run. Iteration
`0` trains without pseudo-labels, then generates them for iteration `1`.

```yaml
start_iteration: 0
iterations: 5
teacher_checkpoint: auto
resume_output_dir: auto
resume_run: auto
```

These are the defaults. Remove conflicting overrides in inherited/debug configs.
No teacher or previous logging state is loaded; metric counters start at zero.

#### 2. Resume an existing run at iteration N

Here `N = 2`: read the teachers from iteration `1`, regenerate their pseudo-labels,
and train iterations `2` through `4`. Reuse the output folder and saved W&B run.

```yaml
start_iteration: 2
iterations: 5
teacher_checkpoint: auto
resume_output_dir: "outputs/old_run"
resume_run: auto
```

Teacher selection is restored exactly from `iteration/1/checkpoints/teacher_checkpoints.json`,
including saved weight averages. Missing/invalid selections or missing files raise
an error; resume does not rebuild the ensemble from Top-k/SWA catalogs.
The W&B identity is stored as `wandb.run_path` in `.hydra/config.yaml`.
Explicit checkpoints can override the automatic teacher selection. For older
runs without `wandb.run_path`, automatic resume raises an error; set
`resume_run: "entity/project/run_id"` once. There is no fallback to the old
config's `train.resume_run`. A missing config also prevents automatic W&B resume.
W&B login and access are required.

#### 3. Start Noisy Student training from a supervised checkpoint

Generate pseudo-label version `0` from the supervised teacher, then train five
student iterations (`1` through `5`) in a new folder and W&B run.

```yaml
start_iteration: 1
iterations: 6
teacher_checkpoint: "outputs/supervised/baseline/checkpoints/best.pt"
resume_output_dir: auto
resume_run: auto
```

`resume_output_dir: auto` or `null` selects a new experiment. Replace `best.pt`
with your actual checkpoint and use a unique experiment name per ablation.
**The checkpoint only generates pseudo-labels;
it does not initialize student weights.** Students use the configured
`build_model(cfg)` initialization. Model/data settings must match the teacher.

#### 4. Start a new experiment with several teacher checkpoints

Use the supplied ensemble to generate pseudo-label version `1`, then train
iterations `2` through `4`. Setting `resume_output_dir: auto` creates a new
folder and, with `resume_run: auto`, a new W&B run.

```yaml
start_iteration: 2
iterations: 5
teacher_checkpoint:
  - "outputs/source/iteration/1/checkpoints/top_k_average.pt"
  - "outputs/source/iteration/1/checkpoints/swa_average.pt"
resume_output_dir: auto
resume_run: auto
```

Explicit checkpoint-file lists are used as given for the initial pseudo-label generation.
After each newly trained student iteration, its own checkpoints are selected using
`train.teacher` and `train.checkpoint`. Folder lists, mixed folder/file lists and
explicit JSON manifest paths are rejected. A single folder supplied as `teacher_checkpoint`
also starts a new experiment when `resume_output_dir` is `auto` or `null`.
To resume instead, set `resume_output_dir` to the old run folder and leave
`teacher_checkpoint: auto`, as in example 2. Alternatively, `resume_output_dir:
checkpoint` infers the common source run from explicit checkpoints. Files from
different runs can be combined when starting a new experiment if their resolved
filenames are distinct. Repeated paths to the same checkpoint remain allowed.

> NOTE
> - **Teacher selection:** Supply one folder or explicit checkpoint files. Folder
>   input for a new experiment uses `train.teacher`/`train.checkpoint` and the available
>   Top-k/SWA catalogs, then writes a new `teacher_checkpoints.json` in the new run.
>   Fewer than k entries require `checkpoint_potential_maximum_k.json` confirming a short
>   training run (Top-k) or an early best epoch (SWA); insufficient retention raises an error.
>   `save_top_k: -1` requires availability metadata confirming that all eligible
>   checkpoints are listed; a truncated or unverified catalog raises an error.
>   Resume instead reuses the saved `teacher_checkpoints.json` exactly. For legacy
>   runs without the required manifest/catalogs, supply checkpoint files explicitly.
>   An iteration/checkpoint folder selects that iteration directly.
> - **Imported weight averages and BatchNorm:** Starting at iteration 1 from a
>   supervised baseline calibrates newly computed averages on labeled training data,
>   matching the baseline. Resuming in the same output folder restores saved averages
>   unchanged. Neither workflow is affected by the following limitation: a new
>   experiment starting at `start_iteration > 1` from a folder with `top_k_mode: average`
>   or `swa_mode: average` currently recalibrates BatchNorm using only labeled data,
>   omitting the source student's pseudo-labeled training data. This can change
>   predictions; the effect depends on the model and data. To preserve existing
>   averages in that case, supply their checkpoint files explicitly.
> - **Logging:** Continuations always append; old points and duplicate X values
>   remain. There is no rewind or continuation into a different W&B run.
>   Counters are reconstructed exclusively from the same W&B run at the requested
>   iteration boundary. New experiments start at zero. History must already be
>   synced and accessible through the W&B API; no local counter fallback exists.
> - **Backups:** Existing output/temp iteration folders are renamed to
>   `<iteration>_backup_<timestamp>` before reuse, including `N - 1` before
>   pseudo-label generation. Teacher paths are adjusted; later folders are backed
>   up when reached. Backups are not deleted automatically.
> - **Config checks:** Only continuations compare configs, and only the old
>   config in the same output folder is read. Its `.hydra` is copied to
>   `.hydra_backup_<timestamp>` before replacement. Changed common settings abort;
>   added/removed keys warn. Resume fields, experiment metadata, and output/temp
>   paths may change. Legacy configs without resolved snapshots are compared
>   as expressions, so historical environment values cannot be verified.
> - **New experiments:** Save only their current config; teacher-run configs
>   are not read or copied. Occupied output folders are rejected to prevent
>   accidental overwrites. Use a unique experiment name.
> - **Config records:** A continuation requires `.hydra/config.yaml`. Automatic
>   W&B lookup also requires `wandb.run_path`; legacy runs can supply the actual
>   ID explicitly. A conflicting saved ID is rejected. Successful starts save
>   Hydra metadata and `config.resolved.yaml`; rejected config changes leave the
>   old config active and save the attempted config in the backup.
> - **Resume boundary:** Only whole iterations can be restarted; optimizer and
>   scheduler state are not restored.
> - **Shared data:** Pseudo-label versions under `path.data_dir` are rewritten and
>   are not included in folder backups. Isolate data storage for concurrent
>   experiments that write the same versions.
