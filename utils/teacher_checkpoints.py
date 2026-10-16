"""Resolve explicit teachers or a saved teacher selection without loading weights."""

import json
from pathlib import Path


def _sources(source):
    """Normalize a teacher input into a list without resolving any paths.

    A str or Path is one source, not an iterable of characters/path parts.
    Other inputs are converted with list(source or []), supporting ordinary
    lists and OmegaConf ListConfig values. None or an empty input yields [].

    Returns a new list retaining order and duplicates; repeated checkpoints may
    contribute repeatedly to the existing ensemble behavior. Does not modify
    the input, check files, or load model weights.
    """

    return [source] if isinstance(source, (str, Path)) else list(source or [])


def checkpoint_directory(source, start_iteration):
    """Locate a checkpoint directory from one teacher source.

    Args:
        source: File path, checkpoint directory, iteration directory, or run.
        start_iteration: First iteration to train; a run selects the previous
            iteration, start_iteration - 1.

    Resolve source to an absolute Path, then apply these rules in order:
    1. An existing file or .pt/.pth/.ckpt suffix selects its parent directory.
    2. An existing source/checkpoints directory selects that subdirectory.
    3. An existing source/iteration/<N-1>/checkpoints selects that directory.
    4. Otherwise return the resolved source itself as the candidate directory.

    Example: a run source with start_iteration=3 selects iteration/2/checkpoints.
    Direct iteration/checkpoint input is not checked against start_iteration.
    Returns a Path without creating files. A recognized file suffix permits
    inference for a missing file; resolve_teacher_checkpoints later validates
    actual existence. This helper alone does not guarantee a valid source.
    """

    path = Path(source).resolve()

    if path.is_file() or path.suffix in {".pt", ".pth", ".ckpt"}:
        return path.parent
    
    if (path / "checkpoints").is_dir():
        return path / "checkpoints"
    
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


def _folder_checkpoints(directory):
    """Read the exact teacher selection from a checkpoint folder's manifest.

    Args:
        directory: Checkpoint directory Path, already located by the caller.

    Require teacher_checkpoints.json containing a nonempty list of strings.
    Resolve entries relative to directory and reject paths escaping it.
    Retain order and duplicates; do not infer selection from filenames or
    current Top-K/SWA settings, even when average files are present.

    Returns Paths; the caller checks file existence. Missing/malformed manifests
    or escaping entries raise ValueError (including JSON decoding errors).
    Legacy runs must supply explicit checkpoint files instead of a folder.
    File-read errors propagate. No weights are loaded and no files are changed.
    """

    manifest = directory / "teacher_checkpoints.json"

    if not manifest.is_file():
        raise ValueError(f"{directory} has no teacher_checkpoints.json. Provide an explicit checkpoint file or an explicit checkpoint list for legacy runs.")

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


def resolve_teacher_checkpoints(source, start_iteration):
    """Expand teacher inputs into a validated ordered list of checkpoint Paths.

    Args:
        source: One file/folder or a sequence, including OmegaConf ListConfig.
        start_iteration: Previous-iteration selection for source run folders.
        teacher: Optional legacy argument, retained for caller compatibility.
            Not used: folders must provide an exact saved manifest.

    Explicit existing files are used directly, independently of group settings.
    Directories are located by checkpoint_directory and expanded through
    _folder_checkpoints. Mixed file/folder inputs are concatenated in input
    order; duplicates are deliberately retained. No common run is required
    unless a separate caller requests run-folder inference.

    Returns absolute Paths after verifying every selected entry is a file.
    Raises FileNotFoundError for a missing source or selected checkpoint, and
    ValueError for empty selections or missing/invalid folder manifests.
    Does not mutate cfg, write files, average weights, or initialize students;
    the prediction code consumes the returned teacher ensemble later.
    """

    paths = []

    for item in _sources(source):
        path = Path(item).resolve()

        if path.is_file():
            paths.append(path)
        elif path.is_dir():
            paths.extend(_folder_checkpoints(checkpoint_directory(path, start_iteration)))
        else:
            raise FileNotFoundError(f"Teacher checkpoint or directory not found: {path}")

    if not paths:
        raise ValueError("No teacher checkpoints selected")
    
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"Teacher checkpoint not found: {path}")
        
    return paths


def save_teacher_selection(paths):
    """Write the exact selected ensemble to teacher_checkpoints.json.

    Args:
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
        (directory / "teacher_checkpoints.json").write_text(json.dumps(names, indent=2), encoding="utf-8")
