# Shared settings for ablations A and B

- `base.yaml`: data splits, dataset, sampler, dataloader, loss, training,
  refinement defaults, experiment names, and Hydra output settings.
- `noisy_student.yaml`: common teacher TTA, pseudo-label refinement, iteration,
  model-schedule, teacher checkpoint lookup, and resume settings.
- `unetrpp_*.yaml`: frozen S/M/L model architectures.
- `model_noise_*.yaml`: low/high model noise overrides.
- `model_<size>_<noise>.yaml`: architecture and model-noise combinations.
- `input_noise_*.yaml`: low/medium/high input augmentations.

A and B use the same inheritance structure: local `base.yaml` inherits
`common/base.yaml`; local `noisy_student.yaml` inherits the adjusted local
base and `common/noisy_student.yaml`. Temporary paths and W&B projects live
in the local base configs; Noisy Student output paths and baseline roots
live in the local Noisy Student configs. Each concrete B config selects
its own baseline and teacher aggregation. Checkpoint counts remain
unchanged: A students retain Top-1/SWA-1, A supervised baselines retain
Top-10/SWA-10, and B uses `ablation.k` (default 5).

A and B import shared model and augmentation presets directly. Their model
schedules use `/ablation/common/model_<size>_<noise>` paths. Model and input-noise
presets are defined only here; there are no forwarding configs in A or B.
