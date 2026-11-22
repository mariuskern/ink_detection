"""Check W&B initialization without importing GPU/training dependencies."""

import ast
from collections import defaultdict
from contextlib import nullcontext
from datetime import datetime
import json
from pathlib import Path
import re
import shutil
from tempfile import TemporaryDirectory
from typing import Any
import unittest
from unittest.mock import Mock, patch
import warnings


class WandbResumeTests(unittest.TestCase):
    def setUp(self):
        source = Path(__file__).resolve().parents[2] / "training/logging.py"
        node = next(node for node in ast.parse(source.read_text()).body if isinstance(node, ast.ClassDef) and node.name == "WandbLogger")
        self.wandb = Mock()
        self.wandb.init.return_value.configure_mock(entity="team", project="project", id="id")
        namespace = dict(defaultdict=defaultdict, datetime=datetime, json=json, Path=Path, re=re, shutil=shutil, Any=Any, warnings=warnings, wandb=self.wandb)

        exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), namespace)

        self.Logger = namespace["WandbLogger"]
        self.folder = self.enterContext(TemporaryDirectory())
        self.config = {"path": {"output_dir": self.folder}}
        self.state_path = Path(self.folder) / "wandb_steps.json"

    def test_append_restores_boundary_counters_and_keeps_run_identity(self):
        self.wandb.Api.return_value.run.return_value.scan_history.return_value = iter([ {"train/1_loss": .4, "train/loss": .4, "steps/train_loss_step": 99}, {"train/2_loss": .3, "train/loss": .3, "steps/train_loss_step": 100}, ])
        logger = self.Logger("project", resume_run="team/project/id", start_iteration=2)
        options = self.wandb.init.call_args.kwargs

        self.assertEqual(options["id"], "id")
        self.assertEqual(options["resume"], "must")
        self.assertNotIn("resume_from", options)

        logger.log({"train/loss": .2, "train/2_loss": .2})

        logged = logger.run.log.call_args.args[0]

        self.assertEqual(logged["steps/train_loss_step"], 100)
        self.assertEqual(logged["steps/train_2_loss_step"], 0)

    def test_fresh_run_and_immediate_boundary(self):
        self.Logger("project")

        self.wandb.Api.assert_not_called()
        self.assertNotIn("id", self.wandb.init.call_args.kwargs)
        self.assertEqual(self.Logger._restore_history([{"train/2_loss": 1.}], 2), {})

    def test_local_boundary_restores_all_counters_without_api(self):
        logger = self.Logger("project", config=self.config)
        logger.log({"train/0_loss": .4, "train/loss": .4})
        logger.log({"train/0_loss": .3, "train/loss": .3})
        logger.log({"train/loss_epoch": .35})
        logger.log({"metrics_unlabeled/dice": .8})
        logger.save_iteration(1)
        # Partial progress after the boundary must not affect its counters.
        logger.log({"train/1_loss": .2, "train/loss": .2})

        resumed = self.Logger("project", config=self.config, resume_run="team/project/id", start_iteration=1)
        self.wandb.Api.assert_not_called()
        resumed.log({"train/1_loss": .1, "train/loss": .1})
        logged = resumed.run.log.call_args.args[0]
        self.assertEqual(logged["steps/train_loss_step"], 2)
        self.assertEqual(logged["steps/train_1_loss_step"], 0)
        resumed.log({"metrics_unlabeled/dice": .9})
        self.assertEqual(resumed.run.log.call_args.args[0]["steps/metrics_unlabeled_dice_step"], 1)

    def test_empty_local_boundary_does_not_scan(self):
        self.Logger("project", config=self.config, start_iteration=1)
        self.Logger("project", config=self.config, resume_run="team/project/id", start_iteration=1)
        self.wandb.Api.assert_not_called()

    def test_missing_or_unusable_boundary_falls_back_and_is_cached(self):
        for content in (
            None,
            "invalid JSON",
            json.dumps({"run_path": "team/project/other", "iterations": {"1": {"train/loss": 999}}}),
            json.dumps({"run_path": "team/project/id", "iterations": {"0": {}}}),
            json.dumps({"run_path": "team/project/id", "iterations": {"1": {"train/loss": -1}}}),
            json.dumps({"run_path": "team/project/id", "iterations": {"1": []}}),
        ):
            with self.subTest(content=content):
                if content is None:
                    self.state_path.unlink(missing_ok=True)
                else:
                    self.state_path.write_text(content, encoding="utf-8")
                self.wandb.reset_mock()
                self.wandb.Api.return_value.run.return_value.scan_history.return_value = iter([
                    {"train/0_loss": .4, "train/loss": .4, "steps/train_loss_step": 99},
                    {"train/1_loss": .3, "train/loss": .3, "steps/train_loss_step": 100},
                ])
                with warnings.catch_warnings(record=True):
                    logger = self.Logger("project", config=self.config, resume_run="team/project/id", start_iteration=1)
                self.wandb.Api.return_value.run.return_value.scan_history.assert_called_once()
                self.assertEqual(logger._steps["train/loss"], 100)

                self.wandb.reset_mock()
                resumed = self.Logger("project", config=self.config, resume_run="team/project/id", start_iteration=1)
                self.wandb.Api.assert_not_called()
                self.assertEqual(resumed._steps["train/loss"], 100)

    def test_restart_retains_earlier_boundaries_and_replaces_later_ones(self):
        logger = self.Logger("project", config=self.config)
        logger.log({"train/loss": .4})
        logger.save_iteration(1)
        logger.log({"train/loss": .3})
        logger.save_iteration(2)
        previous_state = self.state_path.read_bytes()

        resumed = self.Logger("project", config=self.config, resume_run="team/project/id", start_iteration=1)
        backups = list(Path(self.folder).glob("wandb_steps_backup_*.json"))
        self.assertEqual(len(backups), 1)
        self.assertEqual(backups[0].read_bytes(), previous_state)
        state = json.loads(self.state_path.read_text(encoding="utf-8"))
        self.assertEqual(set(state["iterations"]), {"0", "1"})
        resumed.log({"train/loss": .2})
        resumed.log({"train/loss": .1})
        resumed.save_iteration(2)

        resumed_again = self.Logger("project", config=self.config, resume_run="team/project/id", start_iteration=2)
        self.assertEqual(resumed_again._steps["train/loss"], 3)
        self.wandb.Api.assert_not_called()

    def test_failed_atomic_replace_keeps_previous_boundary(self):
        logger = self.Logger("project", config=self.config)
        previous_state = self.state_path.read_text(encoding="utf-8")
        logger.log({"train/loss": .4})
        with patch.object(Path, "replace", side_effect=OSError("disk unavailable")):
            with self.assertWarnsRegex(UserWarning, "Cannot save local W&B counters"):
                logger.save_iteration(1)
        self.assertEqual(self.state_path.read_text(encoding="utf-8"), previous_state)

    def test_training_saves_boundaries_after_pseudo_label_metrics(self):
        source = Path(__file__).resolve().parents[2] / "training/noisy_student.py"
        node = next(node for node in ast.parse(source.read_text()).body if isinstance(node, ast.FunctionDef) and node.name == "train_noisy_student")
        cfg = Mock()
        cfg.path.output_dir = self.folder
        cfg.path.tmp_dir = str(Path(self.folder) / "tmp")
        cfg.train.get.side_effect = {"start_iteration": 0}.get
        cfg.train.iterations = 2
        cfg.train.seed = None
        logger = self.Logger("project", config=self.config)

        def fit(cfg, logger, **kwargs):
            logger.log({"train/loss": .4, f"train/{kwargs['iteration']}_loss": .4})
            return Mock(top_checkpoints=[], swa_checkpoints=[])

        def generate(*args, logger=None, **kwargs):
            logger.log({"metrics_unlabeled/dice": .8})

        namespace = dict(
            Path=Path, validate_training_data=Mock(), fit=fit,
            _get_config_for_iteration=lambda cfg, iteration: cfg,
            _iteration_caches=lambda cfg: nullcontext(), get_device=lambda cfg: None,
            save_teacher_selection=Mock(), get_checkpoint_paths=Mock(return_value=[]),
            generate_pseudo_labels=generate,
        )
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), namespace)
        namespace["train_noisy_student"](cfg, logger)

        state = json.loads(self.state_path.read_text(encoding="utf-8"))
        self.assertEqual(state["iterations"]["1"]["metrics_unlabeled/dice"], 1)
        self.assertEqual(state["iterations"]["2"]["metrics_unlabeled/dice"], 2)
        resumed = self.Logger("project", config=self.config, resume_run="team/project/id", start_iteration=1)
        self.assertEqual(resumed._steps["train/loss"], 1)
        self.assertEqual(resumed._steps["metrics_unlabeled/dice"], 1)
        self.wandb.Api.assert_not_called()


if __name__ == "__main__":
    unittest.main()
