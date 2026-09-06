from __future__ import annotations

from collections import defaultdict
from typing import Any

import torch
import wandb


# class WandbLogger():
#     def __init__(self, project: str, name: str | None = None, config: dict | None = None):
#         self.run = wandb.init(
#             project=project,
#             name=name,
#             config=config,
#         )

#     def log(self, metrics: dict[str, Any], step: int | None = None) -> None:
#         if step is None:
#             self.run.log(metrics)
#         else:
#             self.run.log(metrics, step=step)

#     def finish(self) -> None:
#         self.run.finish()


class WandbLogger:
    def __init__(self, project: str, name: str | None = None, config: dict | None = None):
        self.run = wandb.init(
            project=project,
            name=name,
            config=config,
        )

        self._steps: dict[str, int] = defaultdict(int)
        self._defined_metrics: set[str] = set()

    def log(self, metrics: dict[str, Any], step: int | None = None) -> None:
        """
        Log metrics to Weights & Biases.

        If ``step`` is provided, the metrics are logged using the
        explicitly provided W&B step.

        If ``step`` is None, each metric gets its own automatically
        managed step counter.

        Metrics ending in ``_epoch`` are treated as epoch averages.
        They are logged at the same step as the previous metric of
        the same logging name and do not advance the step counter.
        """

        if step is not None:
            self.run.log(metrics, step=step)
            return

        metrics_to_log: dict[str, Any] = {}
        steps_to_increment: set[str] = set()

        for logging_name, value in metrics.items():
            if logging_name.endswith("_epoch"):
                series = logging_name.removesuffix("_epoch")
                current_step = max(self._steps[series] - 1, 0)
            else:
                series = logging_name
                current_step = self._steps[series]
                steps_to_increment.add(series)

            self._define_metric(logging_name, series)

            # if value is None:
            #     continue

            metrics_to_log[logging_name] = value
            metrics_to_log[self._step_name(series)] = current_step

        self.run.log(metrics_to_log)

        for series in steps_to_increment:
            self._steps[series] += 1

    def finish(self) -> None:
        self.run.finish()

    def _define_metric(self, logging_name: str, series: str) -> None:
        """
        Define the W&B step metric for a logging series.
        """

        if logging_name in self._defined_metrics:
            return

        self.run.define_metric(logging_name, step_metric=self._step_name(series))
        self._defined_metrics.add(logging_name)

    @staticmethod
    def _step_name(logging_name: str) -> str:
        """
        Return the W&B metric name used as the X-axis.

        The complete logging name is converted into a valid step
        metric name.

        Examples
        --------
        train/loss
            -> train_loss_step

        train/iteration_0_loss
            -> train_iteration_0_loss_step

        val/loss
            -> val_loss_step
        """

        return f"steps/{logging_name.replace('/', '_')}_step"


class LoggingManager:
    """
    Replacement for Lightning's metric logging.

    Collects metrics during an epoch and logs their epoch averages
    using the provided logger.

    Parameters
    ----------
    logger :
        Logger instance. Supports Lightning-style loggers,
        Weights & Biases, and generic loggers with a ``log`` method.

    iteration : str | None
        Optional iteration identifier for pseudo-label training.
        If set, metrics are logged both as ``stage/name`` and
        ``stage/{iteration}_name``.
    """

    def __init__(self, logger=None, iteration: str | None = None):
        self.logger = logger
        self.iteration = iteration

        # Accumulated weighted sums and weights.
        self._values: dict[str, float] = defaultdict(float)
        self._weights: dict[str, float] = defaultdict(float)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def log(
        self,
        stage: str,
        name: str,
        value,
        *,
        batch_size: int | float = 1.0,
        step: int | None = None,
        on_step: bool = False,
        on_epoch: bool = True,
    ) -> None:
        """
        Log a metric.

        Parameters
        ----------
        stage :
            Training stage, e.g. ``"train"`` or ``"val"``.

        name :
            Metric name, e.g. ``"loss"`` or ``"tile_dice"``.

        value :
            Metric value.

        batch_size :
            Weight used for epoch aggregation. This corresponds to
            Lightning's ``batch_size`` argument.

        step :
            Global training step. Required when ``on_step=True``.

        on_step :
            If True, immediately log the metric.

        on_epoch :
            If True, accumulate the metric for epoch-level logging.
        """

        value = self._to_float(value)

        metric_name = f"{stage}/{name}"
        metric_names = [metric_name]

        if self.iteration is not None:
            metric_names.insert(0, f"{stage}/{self.iteration}_{name}")

        if on_step:
            # if step is None:
            #     raise ValueError("step must be provided when on_step=True")

            self._log_to_backend({name: value for name in metric_names}, step)

        if on_epoch:
            for name in metric_names:
                self._values[name] += value * batch_size
                self._weights[name] += batch_size

    def log_dict(
        self,
        stage: str,
        metrics: dict[str, float | torch.Tensor],
        *,
        batch_size: int | float = 1.0,
        step: int | None = None,
        on_step: bool = False,
        on_epoch: bool = True,
    ) -> None:
        """
        Log multiple metrics at once.

        Parameters
        ----------
        stage :
            Training stage, e.g. ``"train"`` or ``"val"``.

        metrics :
            Dictionary mapping metric names to values.

        batch_size :
            Weight used for epoch aggregation.

        step :
            Global training step. Required when ``on_step=True``.

        on_step :
            If True, immediately log the metrics.

        on_epoch :
            If True, accumulate the metrics for epoch-level logging.
        """

        for name, value in metrics.items():
            self.log(
                stage,
                name,
                value,
                batch_size=batch_size,
                step=step,
                on_step=on_step,
                on_epoch=on_epoch,
            )
    
    def log_once(
        self,
        stage: str,
        name: str,
        value,
        *,
        step: int | None = None,
    ) -> None:
        """
        Log a metric immediately without accumulating it for the epoch.

        The metric is logged exactly once and is not included in
        ``compute()`` or ``log_epoch()``.
        """

        value = self._to_float(value)

        metric_name = f"{stage}/{name}"
        metrics = {metric_name: value}

        if self.iteration is not None:
            metrics[f"{stage}/{self.iteration}_{name}"] = value

        self._log_to_backend(metrics, step)

    def log_dict_once(
        self,
        stage: str,
        metrics: dict[str, float | torch.Tensor],
        *,
        step: int | None = None,
    ) -> None:
        """
        Log multiple metrics at once.

        Parameters
        ----------
        stage :
            Training stage, e.g. ``"train"`` or ``"val"``.

        metrics :
            Dictionary mapping metric names to values.

        step :
            Global training step. Required when ``on_step=True``.
        """

        for name, value in metrics.items():
            self.log_once(
                stage,
                name,
                value,
                step=step,
            )

    def log_epoch(self, step: int | None = None) -> dict[str, float]:
        """
        Log all accumulated epoch metrics and reset the accumulator.

        Returns
        -------
        dict[str, float]
            The computed epoch metrics.
        """

        metrics = self.compute()

        epoch_metrics = {f"{name}_epoch": value for name, value in metrics.items()}

        if epoch_metrics:
            self._log_to_backend(epoch_metrics, step)

        self.reset()

        return epoch_metrics

    def compute(self) -> dict[str, float]:
        """
        Compute the weighted epoch averages without resetting them.
        """

        return {name: self._values[name] / self._weights[name] for name in self._values if self._weights[name] > 0}

    def reset(self) -> None:
        """
        Reset all accumulated metrics.
        """

        self._values.clear()
        self._weights.clear()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _log_to_backend(
        self,
        metrics: dict[str, float],
        step: int | None,
    ) -> None:
        """
        Send metrics to the configured logger.
        """

        if self.logger is None:
            return

        self.logger.log(metrics, step=step)

    @staticmethod
    def _to_float(value) -> float:
        """
        Convert tensors and scalar values to Python floats.
        """

        if isinstance(value, torch.Tensor):
            return value.detach().float().item()

        return float(value)
