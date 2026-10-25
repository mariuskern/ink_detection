from __future__ import annotations

import operator
from pathlib import Path

import torch


class CheckpointManager:
    """
    Tracks monitored top-k checkpoints and optionally keeps the
    latest N checkpoints for SWA.

    Parameters
    ----------
    dirpath : Path | str
        Directory where checkpoints are saved.

    monitor : str
        Metric key to monitor, e.g. ``"val/loss"``.

    mode : str
        ``"min"`` or ``"max"``.

    save_top_k : int
        Number of best checkpoints to keep.

        - ``0``: do not save monitored checkpoints
        - ``1``: keep only the best checkpoint
        - ``>1``: keep the best k checkpoints
        - ``-1``: save all checkpoints

    save_last : bool
        If ``True``, save ``last.pt``.

    save_last_every_epoch : bool
        If ``True``, save ``last.pt`` after every epoch.

    filename_template : str
        Python format string for checkpoint filenames.

    every_n_epochs : int
        Only save monitored/SWA checkpoints every n epochs.

    swa_save_k : int
        Number of most recent checkpoints to keep for SWA.
        ``0`` disables SWA checkpoint storage.
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
        swa_save_k: int = 0,
        model_config: dict | None = None,
    ):
        self.model_config = model_config
        self.dirpath = Path(dirpath)
        self.dirpath.mkdir(parents=True, exist_ok=True)

        if every_n_epochs < 1:
            raise ValueError(f"every_n_epochs must be >= 1, got {every_n_epochs}")

        if save_top_k < -1:
            raise ValueError(f"save_top_k must be >= -1, got {save_top_k}")

        if swa_save_k < 0:
            raise ValueError(f"swa_save_k must be >= 0, got {swa_save_k}")

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
        self.swa_save_k = swa_save_k

        # List of (score, path), sorted best-first.
        self._top_k: list[tuple[float, Path]] = []

        # Checkpoints selected for SWA: the current best and its k-1 predecessors.
        self._swa_checkpoints: list[tuple[int, Path]] = []

        # Rolling history of the most recent k checkpoints, oldest-first.
        self._checkpoint_history: list[tuple[int, Path]] = []

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
        - ``last.pt``
        - a monitored checkpoint
        - one or more top-k checkpoints
        """

        is_checkpoint_epoch = (epoch + 1) % self.every_n_epochs == 0

        # --------------------------------------------------------------
        # Save last.pt
        # --------------------------------------------------------------

        if self.save_last and (self.save_last_every_epoch or is_checkpoint_epoch):
            self._save_last(
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
            )

        if not is_checkpoint_epoch:
            return

        # --------------------------------------------------------------
        # Get monitored metric
        # --------------------------------------------------------------

        score = metrics.get(self.monitor)

        if score is None:
            return

        score = float(score)

        # filename = self.filename_template.format(epoch=epoch, val_loss=score) + ".ckpt"
        filename = self.filename_template.format(epoch=epoch, val_loss=score) + ".pt"
        filepath = self.dirpath / filename

        # --------------------------------------------------------------
        # Determine whether checkpoint is needed
        # --------------------------------------------------------------

        should_save_top_k = self._should_save_top_k(score)
        should_save_swa = self.swa_save_k > 0

        if not should_save_top_k and not should_save_swa:
            return

        is_new_best = self._best_score is None or self._is_better(score, self._best_score)
        
        if is_new_best:
            self._best_score = score
            self._best_path = filepath

        # --------------------------------------------------------------
        # Save checkpoint once
        # --------------------------------------------------------------
        
        self._save_checkpoint(
            filepath=filepath,
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
        )

        paths_to_delete: list[Path] = []

        # --------------------------------------------------------------
        # Add to SWA history
        # --------------------------------------------------------------

        if should_save_swa:
            self._checkpoint_history.append((epoch, filepath))

            while len(self._checkpoint_history) > self.swa_save_k:
                _, old_path = self._checkpoint_history.pop(0)
                paths_to_delete.append(old_path)

                # # Only delete if the checkpoint is not needed by top-k.
                # if not self._is_path_in_top_k(old_path) and not self._is_path_in_swa(old_path):
                #     if old_path.exists():
                #         old_path.unlink()

            # is_new_best = self._best_score is None or self._is_better(score, self._best_score)
            
            if is_new_best:
                # for _, old_path in self._swa_checkpoints:
                #     if not self._is_path_in_history(old_path) and not self._is_path_in_top_k(old_path):
                #         if old_path.exists():
                #             old_path.unlink()
                paths_to_delete.extend(path for _, path in self._swa_checkpoints)
                self._swa_checkpoints = self._checkpoint_history.copy()

        # --------------------------------------------------------------
        # Add to top-k
        # --------------------------------------------------------------

        if should_save_top_k:
            self._top_k.append((score, filepath))
            self._sort_top_k()

            while len(self._top_k) > self.save_top_k:
                _, old_path = self._top_k.pop()
                paths_to_delete.append(old_path)

                # # Only delete if it is not needed by SWA.
                # if not self._is_path_in_swa(old_path) and not self._is_path_in_history(old_path):
                #     if old_path.exists():
                #         old_path.unlink()

            # self._update_best()
        
        # --------------------------------------------------------------
        # Delete old checkpoints that are no longer needed by either top-k or SWA
        # --------------------------------------------------------------

        for old_path in paths_to_delete:
            if not self._is_path_referenced(old_path):
                if old_path.exists():
                    old_path.unlink()

    def clear(self) -> None:
        """
        Delete all checkpoints not referenced by top-k or SWA
        """

        checkpoints_to_delete = self._checkpoint_history.copy()
        self._checkpoint_history.clear()

        for _, path in checkpoints_to_delete:
            if not self._is_path_referenced(path):
                if path.exists():
                    path.unlink()

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
    def swa_model_paths(self) -> list[Path]:
        """
        Checkpoint-Pfade, die für SWA gespeichert wurden, sortiert vom ältesten zum neuesten.
        """

        return [path for _, path in self._swa_checkpoints]

    @property
    def swa_model_scores(self) -> list[float]:
        """
        Scores der gespeicherten SWA-Modelle.
        """
        
        return [score for score, _ in self._swa_checkpoints]

    @property
    def last_model_path(self) -> Path | None:
        # path = self.dirpath / "last.ckpt"
        path = self.dirpath / "last.pt"
        return path if path.exists() else None

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _should_save_top_k(self, score: float) -> bool:
        if self.save_top_k == 0:
            return False

        if self.save_top_k == -1:
            return True

        return len(self._top_k) < self.save_top_k or self._is_better(score, self._top_k[-1][0])

    def _is_path_in_top_k(self, path: Path) -> bool:
        return any(saved_path == path for _, saved_path in self._top_k)

    def _is_path_in_swa(self, path: Path) -> bool:
        return any(saved_path == path for _, saved_path in self._swa_checkpoints)

    def _is_path_in_history(self, path: Path) -> bool:
        return any(saved_path == path for _, saved_path in self._checkpoint_history)

    def _is_path_referenced(self, path: Path) -> bool:
        return self._is_path_in_top_k(path) or self._is_path_in_swa(path) or self._is_path_in_history(path) or self._best_path == path

    def _sort_top_k(self) -> None:
        """
        Sort checkpoints from best to worst.
        """

        self._top_k.sort(key=lambda item: item[0], reverse=self._sort_reverse)

    # def _update_best(self) -> None:
    #     """
    #     Update best_model_path and best_model_score based on the
    #     currently retained checkpoints.
    #     """

    #     if not self._top_k:
    #         self._best_score = None
    #         self._best_path = None
    #         return

    #     best_score, best_path = self._top_k[0]

    #     self._best_score = best_score
    #     self._best_path = best_path

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

        if self.model_config is not None:
            state["model_config"] = self.model_config

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
        Save/overwrite last.pt.
        """

        # last_path = self.dirpath / "last.ckpt"
        last_path = self.dirpath / "last.pt"

        self._save_checkpoint(
            filepath=last_path,
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
        )
