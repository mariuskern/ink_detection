from __future__ import annotations

import operator
from pathlib import Path

import torch


class CheckpointManager:
    """
    Replacement for Lightning's ModelCheckpoint.

    Tracks a monitored metric, keeps the top-k checkpoints, and
    optionally saves a last.ckpt either every epoch or only at
    regular checkpoint intervals.

    Parameters
    ----------
    dirpath : Path | str
        Directory where checkpoints are saved.

    monitor : str
        Metric key to monitor, e.g. ``"val/loss"``.

    mode : str
        ``"min"`` or ``"max"``. Determines whether lower or higher
        metric values are considered better.

    save_top_k : int
        Number of best checkpoints to keep.

        - ``0``: do not save monitored checkpoints
        - ``1``: keep only the best checkpoint
        - ``>1``: keep the best k checkpoints
        - ``-1``: save all checkpoints

    save_last : bool
        If ``True``, save ``last.ckpt``.

    save_last_every_epoch : bool
        If ``True``, ``last.ckpt`` is saved after every epoch,
        independently of ``every_n_epochs``.

        If ``False``, ``last.ckpt`` is only saved at regular
        checkpoint intervals defined by ``every_n_epochs``.

    filename_template : str
        Python format string for checkpoint filenames.
        Available keys are ``epoch`` and ``val_loss``.

    every_n_epochs : int
        Only consider monitored checkpoints every n epochs.
        Epochs are zero-based internally, but the interval is
        counted as human-readable epochs.

    Example
    -------
    ``every_n_epochs=5`` saves monitored checkpoints after epochs
    4, 9, 14, ...

    With ``save_last_every_epoch=True``, ``last.ckpt`` is still
    updated after epochs 0, 1, 2, 3, 4, 5, ...
    """

    def __init__(
        self,
        dirpath: Path | str,
        monitor: str,
        mode: str,
        save_top_k: int,
        save_last: bool,
        filename_template: str,
        every_n_epochs: int = 1,
        save_last_every_epoch: bool = False,
    ):
        self.dirpath = Path(dirpath)
        self.dirpath.mkdir(parents=True, exist_ok=True)

        if every_n_epochs < 1:
            raise ValueError(f"every_n_epochs must be >= 1, got {every_n_epochs}")

        if save_top_k < -1:
            raise ValueError(f"save_top_k must be >= -1, got {save_top_k}")

        if mode == "min":
            self._is_better = operator.lt
            self._sort_reverse = False
        elif mode == "max":
            self._is_better = operator.gt
            self._sort_reverse = True
        else:
            raise ValueError(f"mode must be 'min' or 'max', got {mode!r}")

        self.monitor = monitor
        self.save_top_k = save_top_k
        self.save_last = save_last
        self.save_last_every_epoch = save_last_every_epoch
        self.filename_template = filename_template
        self.every_n_epochs = every_n_epochs

        # List of (score, path), sorted best-first.
        self._top_k: list[tuple[float, Path]] = []

        self._best_score: float | None = None
        self._best_path: Path | None = None

    def step(
        self,
        epoch: int,
        metrics: dict[str, float],
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer | None = None,
        scheduler=None,
    ) -> None:
        """
        Call at the end of every validation epoch.

        Depending on the configuration, this may save:
        - ``last.ckpt``
        - a monitored checkpoint
        - one or more top-k checkpoints
        """

        is_checkpoint_epoch = (epoch + 1) % self.every_n_epochs == 0

        # --------------------------------------------------------------
        # Save last.ckpt
        # --------------------------------------------------------------
        if self.save_last and (self.save_last_every_epoch or is_checkpoint_epoch):
            self._save_last(
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
            )

        # --------------------------------------------------------------
        # Top-k checkpoints are only considered at checkpoint intervals
        # --------------------------------------------------------------
        if not is_checkpoint_epoch:
            return

        # save_top_k=0 means no monitored checkpoints.
        if self.save_top_k == 0:
            return

        score = metrics.get(self.monitor)

        if score is None:
            return

        score = float(score)

        filename = self.filename_template.format(epoch=epoch, val_loss=score) + ".ckpt"
        filepath = self.dirpath / filename

        # --------------------------------------------------------------
        # save_top_k=-1 -> save every checkpoint
        # --------------------------------------------------------------
        if self.save_top_k == -1:
            self._save_checkpoint(
                filepath=filepath,
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
            )

            self._top_k.append((score, filepath))
            self._sort_top_k()

            self._update_best()

            return

        # --------------------------------------------------------------
        # save_top_k > 0
        # --------------------------------------------------------------
        should_save = len(self._top_k) < self.save_top_k or self._is_better(score, self._top_k[-1][0])

        if not should_save:
            return

        # Save candidate checkpoint.
        self._save_checkpoint(
            filepath=filepath,
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
        )

        self._top_k.append((score, filepath))
        self._sort_top_k()

        # Remove checkpoints outside top-k.
        while len(self._top_k) > self.save_top_k:
            _, old_path = self._top_k.pop()

            if old_path.exists():
                old_path.unlink()

        self._update_best()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def best_model_path(self) -> Path | None:
        """
        Path to the currently best checkpoint.
        """
        
        return self._best_path

    @property
    def best_model_score(self) -> float | None:
        """
        Score of the currently best checkpoint.
        """

        return self._best_score

    @property
    def top_k_model_paths(self) -> list[Path]:
        """
        Checkpoint-Pfade, sortiert vom besten zum schlechtesten Modell.
        """

        return [path for _, path in self._top_k]


    @property
    def top_k_model_scores(self) -> list[float]:
        """
        Scores der gespeicherten Top-k-Modelle.
        """
        
        return [score for score, _ in self._top_k]


    @property
    def last_model_path(self) -> Path | None:
        path = self.dirpath / "last.ckpt"
        return path if path.exists() else None

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _sort_top_k(self) -> None:
        """
        Sort checkpoints from best to worst.
        """

        self._top_k.sort(key=lambda item: item[0], reverse=self._sort_reverse)

    def _update_best(self) -> None:
        """
        Update best_model_path and best_model_score based on the
        currently retained checkpoints.
        """

        if not self._top_k:
            self._best_score = None
            self._best_path = None
            return

        best_score, best_path = self._top_k[0]

        self._best_score = best_score
        self._best_path = best_path

    def _save_checkpoint(
        self,
        filepath: Path,
        epoch: int,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer | None,
        scheduler,
    ) -> None:
        """
        Save a training checkpoint.
        """

        state = {
            "epoch": epoch,
            "state_dict": model.state_dict(),
        }

        if optimizer is not None:
            state["optimizer_state_dict"] = optimizer.state_dict()

        if scheduler is not None:
            state["scheduler_state_dict"] = scheduler.state_dict()

        torch.save(state, filepath)

    def _save_last(
        self,
        epoch: int,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer | None,
        scheduler,
    ) -> None:
        """
        Save/overwrite last.ckpt.
        """

        last_path = self.dirpath / "last.ckpt"

        self._save_checkpoint(
            filepath=last_path,
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
        )



# """
# Replacement for Lightning's ``ModelCheckpoint`` callback.

# Tracks a monitored metric, keeps the top-k checkpoints, and optionally
# saves a ``last.ckpt`` every epoch.
# """

# from __future__ import annotations

# import operator
# from pathlib import Path

# import torch


# class CheckpointManager:
#     """
#     Manages checkpoint saving with top-k and last-checkpoint logic,
#     mirroring Lightning's ``ModelCheckpoint``.

#     Parameters
#     ----------
#     dirpath : Path | str
#         Directory where checkpoints are saved.
#     monitor : str
#         Metric key to monitor (e.g. ``"val/loss"``).
#     mode : str
#         ``"min"`` or ``"max"``—whether lower or higher is better.
#     save_top_k : int
#         Number of best checkpoints to keep.
#     save_last : bool
#         If ``True``, always save a ``last.ckpt`` at the end of each
#         checkpoint step.
#     filename_template : str
#         Python format-string for the checkpoint filename. Available
#         keys: ``epoch``, ``val_loss`` (the monitored value).
#     every_n_epochs : int
#         Only consider saving every *n* epochs.
#     """

#     def __init__(
#         self,
#         dirpath: Path | str,
#         monitor: str,
#         mode: str,
#         save_top_k: int,
#         save_last: bool,
#         filename_template: str,
#         every_n_epochs: int = 1,
#     ):
#         self.dirpath = Path(dirpath)
#         self.dirpath.mkdir(parents=True, exist_ok=True)

#         self.monitor = monitor
#         self.save_top_k = save_top_k
#         self.save_last = save_last
#         self.filename_template = filename_template
#         self.every_n_epochs = every_n_epochs

#         if mode == "min":
#             self._is_better = operator.lt
#             self._worst = float("inf")
#         elif mode == "max":
#             self._is_better = operator.gt
#             self._worst = float("-inf")
#         else:
#             raise ValueError(f"mode must be 'min' or 'max', got {mode!r}")

#         # List of (score, path) tuples, sorted best-first.
#         self._top_k: list[tuple[float, Path]] = []
#         self._best_score: float | None = None
#         self._best_path: Path | None = None

#     # ------------------------------------------------------------------
#     # Public API
#     # ------------------------------------------------------------------

#     def step(
#         self,
#         epoch: int,
#         metrics: dict[str, float],
#         model: torch.nn.Module,
#         optimizer: torch.optim.Optimizer | None = None,
#         scheduler=None,
#     ):
#         """
#         Call at the end of each validation epoch. Saves checkpoints
#         when appropriate.
#         """
#         if self.every_n_epochs > 1 and (epoch + 1) % self.every_n_epochs != 0:
#             # Still save *last* if requested.
#             if self.save_last:
#                 self._save_last(epoch, model, optimizer, scheduler)
#             return

#         score = metrics.get(self.monitor)
#         if score is None:
#             if self.save_last:
#                 self._save_last(epoch, model, optimizer, scheduler)
#             return

#         # Determine filename.
#         filename = self.filename_template.format(
#             epoch=epoch, val_loss=score,
#         ) + ".ckpt"
#         filepath = self.dirpath / filename

#         # Always save last.
#         if self.save_last:
#             self._save_last(epoch, model, optimizer, scheduler)

#         # Top-k logic.
#         if self.save_top_k == 0:
#             return

#         should_save = (
#             len(self._top_k) < self.save_top_k
#             or self._is_better(score, self._top_k[-1][0])
#         )

#         if should_save:
#             self._save_checkpoint(filepath, epoch, model, optimizer, scheduler)
#             self._top_k.append((score, filepath))
#             # Sort: best first.
#             self._top_k.sort(
#                 key=lambda x: x[0],
#                 reverse=(self._worst == float("-inf")),  # max → descending
#             )
#             # Prune.
#             while len(self._top_k) > self.save_top_k:
#                 _, old_path = self._top_k.pop()
#                 if old_path.exists():
#                     old_path.unlink()

#         # Update global best.
#         if self._best_score is None or self._is_better(score, self._best_score):
#             self._best_score = score
#             self._best_path = self._top_k[0][1] if self._top_k else filepath

#     @property
#     def best_model_path(self) -> Path | None:
#         return self._best_path

#     @property
#     def best_model_score(self) -> float | None:
#         return self._best_score

#     # ------------------------------------------------------------------
#     # Internal
#     # ------------------------------------------------------------------

#     def _save_checkpoint(self, filepath, epoch, model, optimizer, scheduler):
#         state = {
#             "epoch": epoch,
#             "state_dict": model.state_dict(),
#         }
#         if optimizer is not None:
#             state["optimizer_state_dict"] = optimizer.state_dict()
#         if scheduler is not None:
#             state["scheduler_state_dict"] = scheduler.state_dict()

#         torch.save(state, filepath)

#     def _save_last(self, epoch, model, optimizer, scheduler):
#         last_path = self.dirpath / "last.ckpt"
#         self._save_checkpoint(last_path, epoch, model, optimizer, scheduler)
