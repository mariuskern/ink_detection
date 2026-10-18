"""Archive Hydra metadata and validate configuration before resuming training."""

from datetime import datetime
from pathlib import Path
import shutil
import warnings

from omegaconf import OmegaConf, open_dict
from utils.teacher_checkpoints import teacher_run_directory


# Ignore historical control fields as well, so older run snapshots remain comparable.
RESUME_FIELDS = {
    "train.iterations", "train.start_iteration", "train.teacher_checkpoint", "train.logging_state",
    "train.resume_run", "train.resume_mode", "train.resume_output_dir", "experiment.name", "experiment.jobid",
    "path.output_dir", "path.tmp_dir", "path.data_dir", "wandb.run_path",
}


def resolve_training_output_dir(previous, current, checkpoint=None, start_iteration=1):
    """Resolve the training output path for Hydra before main is entered.

    Args:
        previous: train.resume_output_dir: "auto", None, an explicit path,
            or "checkpoint". The name refers to the option, not a saved path.
        current: Regular experiment path, assembled from OUTPUT_DIR, training
            name, model name, and experiment.name by the YAML expression.
        checkpoint: Explicit teacher source used only for "checkpoint" mode.
        start_iteration: Iteration used to locate teachers in a source run.

    "auto" and None return current. An explicit path is returned unchanged;
    "checkpoint" delegates to teacher_run_directory and returns its path as a
    string. For example, a teacher at run/iteration/2/checkpoints/model.pt
    produces the run root. Multiple teachers must have a common run root.

    Returns the selected path; does not create directories or modify cfg.
    Raises ValueError for missing/automatic teachers in "checkpoint" mode or
    a nonstandard/ambiguous source layout. This must run before Hydra chooses
    hydra.run.dir, rather than changing that destination inside main.
    """

    if previous == "auto":
        previous = None

    if previous != "checkpoint":
        return previous or current
    
    if not checkpoint or checkpoint == "auto":
        raise ValueError("resume_output_dir=checkpoint requires train.teacher_checkpoint")
    
    return str(teacher_run_directory(checkpoint, start_iteration))


def register_config_resolvers():
    """Register the training_output_dir resolver in OmegaConf's global registry.

    YAML expressions of the form ${training_output_dir:...} call
    resolve_training_output_dir with the evaluated positional arguments.
    Call this before Hydra evaluates its output-directory interpolation.

    An existing registration with the same name is replaced, allowing repeated
    setup in tests or the same process. Returns None and does not evaluate a
    configuration, create folders, or initialize W&B.
    """
    
    OmegaConf.register_new_resolver("training_output_dir", resolve_training_output_dir, replace=True)


def prepare_teacher_source(cfg):
    """Resolve automatic teacher input by modifying cfg.train.teacher_checkpoint.

    Only applies to Noisy Student training. Explicit teacher files, lists, or
    folders are left unchanged. For teacher_checkpoint=None or "auto":

    - If start_iteration > 0 and resume_output_dir is an explicit folder path,
      require that folder to exist and use it as the teacher source.
    - Otherwise, set teacher_checkpoint to None (no teacher source).

    Example: start_iteration=3, teacher_checkpoint="auto", and
    resume_output_dir="outputs/old_run" replaces teacher_checkpoint with
    "outputs/old_run". Later, resolve_teacher_checkpoints selects the teacher
    files from iteration 2. This function does not select or load any weights.

    At iteration 0, automatic teacher input becomes None for a fresh start.
    At a later iteration, None without an explicit resume folder also remains
    None; subsequent validation rejects the missing teacher. The output values
    "auto" and "checkpoint" are selection modes, not explicit folder paths.

    Call before teacher validation and snapshot creation so training and saved
    configs use the same concrete source. Returns None; updates cfg in place.
    Raises FileNotFoundError if the inferred explicit resume folder is missing.
    """

    if cfg.train.name != "noisy_student":
        return
    
    source = cfg.train.get("teacher_checkpoint")

    if source is None or source == "auto":
        folder = cfg.train.get("resume_output_dir")

        if cfg.train.get("start_iteration", 0) > 0 and folder not in {None, "auto", "checkpoint"}:
            if not Path(folder).is_dir():
                raise FileNotFoundError(f"Resume output directory not found: {folder}")
            
            cfg.train.teacher_checkpoint = folder
        else:
            cfg.train.teacher_checkpoint = None


def is_resuming(cfg):
    """Return whether an old output folder was requested, independently of iteration.

    auto/None denotes a new experiment, including teacher-based ablations.
    An explicit folder or checkpoint mode denotes continuation in that folder.
    """
    
    return cfg.train.get("resume_output_dir", "auto") not in {None, "auto"}


def resolve_resume_run(cfg):
    """Validate the experiment mode and return its existing W&B identity or None.

    New experiments (resume_output_dir=auto/None) require resume_run=auto/None
    and always create a new W&B run. Continuations require the old output
    folder's .hydra/config.yaml and must append to the same W&B run.

    Automatic identity lookup requires wandb.run_path. A legacy config without
    that field may use an explicit entity/project/id. If an identity is saved,
    an explicit ID must match it. None is not allowed for continuations.
    No files are modified and no W&B connection is made. Contradictions or
    missing resume metadata raise ValueError before logging or backups begin.
    """

    requested = cfg.train.get("resume_run", "auto")

    if not is_resuming(cfg):
        if requested not in {None, "auto"}:
            raise ValueError("A new output folder requires a new W&B run; use resume_run=auto or null")
        return None

    config_file = Path(cfg.path.output_dir).resolve() / ".hydra/config.yaml"

    if not config_file.is_file():
        raise ValueError(f"Continuation requires the old config: {config_file}")
    
    if requested is None:
        raise ValueError("Continuation must use the existing W&B run; set resume_run=auto or its run ID")
    
    if requested == "auto":
        return _read_wandb_identity(config_file)
    
    saved = OmegaConf.select(OmegaConf.load(config_file), "wandb.run_path")

    if saved and saved != requested:
        raise ValueError(f"resume_run {requested!r} does not match the saved W&B run {saved!r}")
    
    return requested


def _read_wandb_identity(config_file):
    """Read the actual saved W&B identity, with no legacy inference.

    Args:
        config_file: Path to the run's .hydra/config.yaml.

    Read only wandb.run_path, which records the initialized run as
    entity/project/id. train.resume_run is not a substitute: it can identify
    a source run rather than the run that produced these outputs.

    Returns the stored nonempty identity unchanged. Raises ValueError when
    the config file or identity is missing, also for legacy runs. The error
    explains how to set train.resume_run explicitly or select a new run.
    YAML parsing/interpolation errors propagate. No files or cfg are modified,
    no W&B connection is made, and run-path syntax is checked by the logger.
    """

    identity = None

    if config_file.is_file():
        identity = OmegaConf.select(OmegaConf.load(config_file), "wandb.run_path")

    if not identity:
        raise ValueError(f"No saved W&B run identity found in {config_file}. Set train.resume_run=entity/project/run_id once. For a new experiment, use resume_output_dir=auto and resume_run=auto.")
    
    return identity


def save_wandb_identity(cfg, hydra_cfg, run):
    """Persist the initialized W&B run's actual identity in cfg and snapshots.

    Args:
        cfg: Mutable task configuration containing the wandb section.
        hydra_cfg: Hydra config whose runtime.output_dir is the selected output.
        run: Initialized W&B run exposing entity, project, and id.

    Set cfg.wandb.run_path to entity/project/id, temporarily allowing a new key
    in a structured config. Rewrite .hydra/config.yaml and config.resolved.yaml
    so future continuations can identify the same run.

    Call after save_training_config and successful W&B initialization: the
    metadata folder must already exist. Returns None; modifies cfg and files
    but makes no W&B API calls and creates no additional backup. Filesystem or
    interpolation errors propagate. The startup snapshot remains available if
    W&B initialization fails before this function is reached.
    """

    with open_dict(cfg.wandb):
        cfg.wandb.run_path = f"{run.entity}/{run.project}/{run.id}"

    directory = Path(hydra_cfg.runtime.output_dir) / ".hydra"

    _write_task_config(cfg, directory, OmegaConf.to_container(cfg, resolve=True))


def compare_configs(old, new, prefix=""):
    """Compare plain config values recursively, ignoring allowed resume fields.

    Args:
        old: Previous dict/list/scalar configuration, not an OmegaConf node.
        new: Current configuration in the same plain-container form.
        prefix: Current key path, initially empty; used internally by recursion.

    Returns (changed, added, removed), each a list of readable strings.
    Changed entries include the path and old/new values; additions/removals
    contain key paths. Example: model.width: 8 -> 16. Paths in RESUME_FIELDS
    are skipped entirely, including any nested content below those paths.

    Dict keys are traversed in sorted order. Equal-length lists are compared
    by index, so order matters; differing lengths produce one change without
    descending into their entries. Scalar type changes also count (True and 1
    are different). Entire new/removed dict branches are reported at their key,
    rather than enumerating every nested key.

    Does not mutate inputs, access files, warn, or raise on differences. The
    caller decides which reported changes should prevent training.
    """

    changed, added, removed = [], [], []

    if prefix in RESUME_FIELDS:
        return changed, added, removed
    
    if isinstance(old, dict) and isinstance(new, dict):
        for key in sorted(old.keys() | new.keys()):
            path = f"{prefix}.{key}" if prefix else str(key)

            if path in RESUME_FIELDS:
                continue

            if key not in old:
                added.append(path)
            elif key not in new:
                removed.append(path)
            else:
                differences = compare_configs(old[key], new[key], path)

                for target, values in zip((changed, added, removed), differences):
                    target.extend(values)
    elif isinstance(old, list) and isinstance(new, list):
        # List order and length affect datasets, augmentations, and model settings.
        if len(old) != len(new):
            changed.append(f"{prefix}: list length {len(old)} -> {len(new)}")
        else:
            for index, (before, after) in enumerate(zip(old, new)):
                differences = compare_configs(before, after, f"{prefix}[{index}]")

                for target, values in zip((changed, added, removed), differences):
                    target.extend(values)
    elif type(old) is not type(new) or old != new:
        changed.append(f"{prefix}: {old!r} -> {new!r}")
    
    return changed, added, removed


def _validate_resume_config(cfg, resolved, previous_directory, backup):
    """Validate an archived previous config against the current task config.

    Args:
        cfg: Current OmegaConf task config, used for unresolved legacy checks.
        resolved: Current plain config with interpolations already evaluated.
        previous_directory: Existing metadata folder containing config.yaml.
        backup: Already-created backup folder for a rejected config snapshot.

    Prefer config.resolved.yaml to compare historical resolved values with
    resolved. For older metadata, warn and compare config.yaml expressions
    against the current unresolved expressions; historical environment values
    cannot be recovered by resolving them now.

    Added/removed keys produce warnings. Changed common values (excluding
    RESUME_FIELDS) save attempted_config.resolved.yaml under backup and raise
    ValueError with the differences. Returns None on success. Existing active
    config files and cfg are never modified; parsing/I/O errors propagate.
    """

    resolved_file = previous_directory / "config.resolved.yaml"

    if resolved_file.is_file():
        old = OmegaConf.to_container(OmegaConf.load(resolved_file), resolve=False)
        new = resolved
    else:
        # Resolving an old expression now cannot recover historical env values.
        warnings.warn("Previous config has no resolved snapshot; comparing unresolved values. Historical environment/interpolation values cannot be verified.")
        old = OmegaConf.to_container(OmegaConf.load(previous_directory / "config.yaml"), resolve=False)
        new = OmegaConf.to_container(cfg, resolve=False)

    changed, added, removed = compare_configs(old, new)

    if added:
        warnings.warn("New configuration keys: " + ", ".join(added))

    if removed:
        warnings.warn("Removed configuration keys: " + ", ".join(removed))

    if changed:
        OmegaConf.save(OmegaConf.create(resolved), backup / "attempted_config.resolved.yaml")
        raise ValueError("Resume configuration differs from previous training:\n" + "\n".join(changed))


def _write_task_config(cfg, directory, resolved):
    """Write task config expressions and their already-resolved counterpart.

    Args:
        cfg: OmegaConf task config, saved to config.yaml without forcing all
            expressions to resolve during that write.
        directory: Existing .hydra Path; this function does not create it.
        resolved: Plain resolved config, saved to config.resolved.yaml.

    Both files are overwritten in place. Returns None, leaves cfg unchanged,
    and performs no backup or validation. Call only after the caller has
    handled preservation/checks. Filesystem and serialization errors propagate.
    """

    OmegaConf.save(cfg, directory / "config.yaml")
    OmegaConf.save(OmegaConf.create(resolved), directory / "config.resolved.yaml")


def save_training_config(cfg, hydra_cfg):
    """Save a fresh config or back up and validate an in-place continuation.

    Args:
        cfg: Task config after teacher normalization and run validation.
        hydra_cfg: Hydra runtime config with output_dir and overrides.task.

    resume_output_dir=auto/None denotes a new experiment: save its own config
    without reading/copying teacher metadata. Reject an occupied destination
    (except the job log created by Hydra) to avoid overwriting another experiment.

    An explicit resume folder or checkpoint mode requires an existing config
    in the actual destination. Copy only this folder's .hydra to a timestamped
    sibling, then compare configurations. Rejected changes retain the old
    config and save the attempted config in the backup. No cross-run search
    or backup occurs. Missing continuation metadata is an error, not a warning.

    On success write config.yaml, config.resolved.yaml, hydra.yaml and
    overrides.yaml. Returns None; does not mutate cfg. Run before W&B startup
    and iteration backups. save_wandb_identity later records the actual run ID.
    """

    root = Path(hydra_cfg.runtime.output_dir)
    directory = root / ".hydra"
    resolved = OmegaConf.to_container(cfg, resolve=True)

    if is_resuming(cfg):
        if not (directory / "config.yaml").is_file():
            raise ValueError(f"Continuation requires the old config: {directory / 'config.yaml'}")
        
        backup = directory.with_name(f".hydra_backup_{datetime.now():%Y%m%d_%H%M%S_%f}")

        shutil.copytree(directory, backup)

        print(f"Preserved previous Hydra configuration: {backup}")

        _validate_resume_config(cfg, resolved, directory, backup)
    elif root.exists() and any(entry.name != f"{OmegaConf.select(hydra_cfg, 'job.name', default='train')}.log" for entry in root.iterdir()):
        raise FileExistsError(f"New experiment output folder is occupied: {root}. Choose a unique experiment.name or resume the old run.")

    directory.mkdir(parents=True, exist_ok=True)

    _write_task_config(cfg, directory, resolved)
    
    OmegaConf.save(OmegaConf.create({"hydra": OmegaConf.to_container(hydra_cfg, resolve=False)}), directory / "hydra.yaml")
    OmegaConf.save(hydra_cfg.overrides.task, directory / "overrides.yaml")
