"""Resolve explicit teachers or a saved teacher selection without loading weights."""

import json
from pathlib import Path


def _sources(source):
    """Accept one folder/file path or an ordered list of explicit checkpoint files.

    Folder lists, mixed folder/file inputs and explicit JSON manifests are not
    supported. Different checkpoint files must have distinct resolved filenames.
    Concrete checkpoint lists retain their order and repeated paths.
    File existence is validated by the resolver, not by this normalization.
    """

    single = isinstance(source, (str, Path))
    values = [source] if single else list(source or [])
    extensions = {".pt", ".pth", ".ckpt"}
    filenames = {}

    for value in values:
        if not isinstance(value, (str, Path)):
            raise ValueError("teacher_checkpoint must be one folder path or explicit checkpoint file(s)")
        
        path = Path(value)

        if not single and (path.is_dir() or path.suffix not in extensions):
            raise ValueError("teacher_checkpoint lists must contain only .pt/.pth/.ckpt files; give a single folder as a scalar path")
        
        if single and not path.is_dir() and (path.is_file() or path.suffix == ".json") and path.suffix not in extensions:
            raise ValueError("teacher_checkpoint must be one folder path or explicit .pt/.pth/.ckpt file(s), not a JSON manifest")

        resolved = path.resolve()
        previous = filenames.setdefault(resolved.name, resolved)
        if previous != resolved:
            raise ValueError(f"teacher_checkpoint files must have distinct filenames: {previous} and {resolved}")
        
    return values


def checkpoint_directory(source, start_iteration=None):
    """Locate a checkpoint directory from one teacher source.

    Args:
        source: File path, checkpoint directory, iteration directory, or run.
        start_iteration: First iteration to train; a run selects the previous
            iteration, start_iteration - 1. Optional for direct file,
            checkpoint-directory or iteration-directory inputs.

    Resolve source to an absolute Path, then apply these rules in order:
    1. An existing file or .pt/.pth/.ckpt suffix selects its parent directory.
    2. An existing source/checkpoints directory selects that subdirectory.
    3. An existing source/iteration/<N-1>/checkpoints selects that directory.
    4. Otherwise return the resolved source itself as the candidate directory.

    Example: a run source with start_iteration=3 selects iteration/2/checkpoints.
    Direct iteration/checkpoint input is not checked against start_iteration.
    Returns a Path without creating files. A recognized file suffix permits
    inference for a missing file; resolve_teacher_checkpoint_groups later validates
    actual existence. This helper alone does not guarantee a valid source.
    """

    path = Path(source).resolve()

    if path.is_file() or path.suffix in {".pt", ".pth", ".ckpt"}:
        return path.parent
    
    if (path / "checkpoints").is_dir():
        return path / "checkpoints"
    
    if start_iteration is None:
        if (path / "iteration").is_dir():
            raise ValueError("start_iteration is required when source is a run directory")
        
        return path

    candidate = path / "iteration" / str(start_iteration - 1) / "checkpoints"

    return candidate if candidate.is_dir() else path


def teacher_run_directory(source, start_iteration):
    """Infer a single common run root from teacher files or folders.

    Args:
        source: One source or a sequence accepted by _sources.
        start_iteration: Previous-iteration selection for run-folder inputs.

    Use checkpoint_directory for each source and require the resulting layout
    <run>/iteration/<numeric iteration>/checkpoints. Return the absolute <run>
    Path only when all sources identify the same root. Different iterations
    within one run are allowed; matching iteration numbers are not enforced.

    Example: run/iteration/2/checkpoints/a.pt and b.pt both identify run.
    Raises ValueError for nonstandard layouts (including ordinary supervised
    folders), empty input, or mixed run roots. Does not verify every checkpoint
    exists, create folders, or load weights; callers validate selected files.
    """

    sources = _sources(source)

    roots = set()

    for item in sources:
        directory = checkpoint_directory(item, start_iteration)

        if (directory.name != "checkpoints" or directory.parent.parent.name != "iteration" or not directory.parent.name.isdigit()):
            raise ValueError("Expected teacher under <run>/iteration/<number>/checkpoints")
        
        roots.add(directory.parents[2])
    
    if len(roots) != 1:
        raise ValueError("Teacher checkpoints must share one run to infer the output folder")
    
    return roots.pop()


def _folder_checkpoints(directory, filename="teacher_checkpoints.json"):
    """Read the exact teacher selection from a checkpoint folder's manifest.

    Args:
        directory: Checkpoint directory Path, already located by the caller.
        filename: JSON file list relative to directory.

    Require the named manifest containing a nonempty list of strings.
    Resolve entries relative to directory and reject paths escaping it.
    Retain order and duplicates; do not infer selection from filenames or
    current Top-K/SWA settings, even when average files are present.

    Returns Paths; the caller checks file existence. Missing/malformed manifests
    or escaping entries raise ValueError (including JSON decoding errors).
    Legacy runs must supply explicit checkpoint files instead of a folder.
    File-read errors propagate. No weights are loaded and no files are changed.
    """

    manifest = directory / filename

    if not manifest.is_file():
        raise ValueError(f"{directory} has no {filename}. Provide an explicit checkpoint file or an explicit checkpoint list for legacy runs.")

    names = json.loads(manifest.read_text(encoding="utf-8"))

    if not isinstance(names, list) or not names or not all(isinstance(name, str) for name in names):
        raise ValueError(f"Invalid teacher selection: {manifest}")

    paths = []

    for name in names:
        selected = (directory / name).resolve()
        # Manifest entries must stay inside the checkpoint directory.
        selected.relative_to(directory.resolve())
        paths.append(selected)
    
    return paths


def resolve_teacher_checkpoint_groups(source, start_iteration=None, teacher=None, checkpoint=None, *, resume=False):
    """Restore saved teachers on resume or select catalogs for a new run.

    Resume reads teacher_checkpoints.json exactly, including saved averages.
    New runs use teacher flags, counts and modes to select Top-k/SWA catalogs;
    an existing teacher_checkpoints.json does not constrain their selection.
    Explicit checkpoint files are used directly in either mode.
    start_iteration is required only to select an iteration from a run directory.
    Fewer than k checkpoints require availability metadata proving an early
    training end (Top-k) or an early best epoch (SWA).
    save_top_k=-1 requires metadata confirming the Top-k catalog is complete.
    Groups contain (paths, mode, catalog name); exact selections have no name.
    """

    paths = [Path(value).resolve() for value in _sources(source)]

    groups = []

    if len(paths) == 1 and paths[0].is_dir():
        directory = checkpoint_directory(paths[0], start_iteration)

        if resume or teacher is None:
            groups.append((_folder_checkpoints(directory), "individual", None))
        else:
            settings = (("top_k", "use_top_k", "save_top_k", "top_k_mode"), ("swa", "use_swa", "swa_save_k", "swa_mode"))

            for name, enabled, count_key, mode_key in settings:
                if not getattr(teacher, enabled, False):
                    continue

                count = getattr(checkpoint, count_key, None)

                if not isinstance(count, int) or count == 0 or count < -1 or (name == "swa" and count == -1):
                    raise ValueError(f"{count_key} must be positive for {name} (save_top_k=-1 selects all)")
                
                mode = getattr(teacher, mode_key, "individual")

                if mode not in {"individual", "average"}:
                    raise ValueError(f"Invalid {mode_key}: {mode}")

                selected = _folder_checkpoints(directory, f"{name}_checkpoints.json")

                if count == -1 or len(selected) < count:
                    metadata = directory / "checkpoint_potential_maximum_k.json"
                    availability = json.loads(metadata.read_text(encoding="utf-8")) if metadata.is_file() else {}
                    available = availability.get(name) if isinstance(availability, dict) else None

                    # Selecting all or fewer than k requires a complete catalog.
                    if type(available) is not int or available != len(selected):
                        if count == -1:
                            raise ValueError(f"Requested top_k=-1 (all checkpoints), but {directory} lists {len(selected)} checkpoints without checkpoint_potential_maximum_k.json confirming a complete Top-k catalog. Provide a complete catalog or request a finite save_top_k.")
                        
                        raise ValueError(f"Requested {name}={count}, but only {len(selected)} checkpoints are listed in {directory}. Fewer checkpoints are allowed only when checkpoint_potential_maximum_k.json confirms an early training end (Top-k) or an early best epoch (SWA).")

                if count != -1:
                    selected = selected[:count] if name == "top_k" else selected[-count:]
                
                groups.append((selected, mode, name))
    elif paths:
        # Explicit files are one fixed ensemble, independent of Top-k/SWA settings.
        groups.append((paths, "individual", None))

    paths = [path for selected, _, _ in groups for path in selected]

    if not paths:
        raise ValueError("No teacher checkpoints selected; enable Top-k and/or SWA for baseline folders")

    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"Teacher checkpoint not found: {path}")
    return groups


def save_teacher_selection(paths, filename="teacher_checkpoints.json"):
    """Write selected checkpoint paths to a named JSON manifest.

    Args:
        filename: Manifest filename; defaults to the exact teacher selection.
        paths: Ordered checkpoint Paths. The first checkpoint's parent is the
            destination directory; every path must be relative to that parent.

    Save filenames/relative paths as an indented JSON list, retaining order and
    duplicates. This records the actual selection after averaging/individual
    selection, allowing folder-based resumes to reuse it after a run is moved
    or its iteration folder is backed up.

    Empty input is a no-op. Otherwise overwrite the manifest in the existing
    directory and return None. Does not create folders or verify file existence.
    Paths outside the first parent raise ValueError; write errors propagate.
    """

    if paths:
        directory = paths[0].parent
        names = [str(path.relative_to(directory)) for path in paths]
        
        (directory / filename).write_text(json.dumps(names, indent=2), encoding="utf-8")
