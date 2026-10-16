"""Check W&B initialization without importing GPU/training dependencies."""

import ast
from collections import defaultdict
from pathlib import Path
import re
from typing import Any
import unittest
from unittest.mock import Mock


class WandbResumeTests(unittest.TestCase):
    def setUp(self):
        source = Path(__file__).resolve().parents[2] / "training/logging.py"
        node = next(node for node in ast.parse(source.read_text()).body if isinstance(node, ast.ClassDef) and node.name == "WandbLogger")
        self.wandb = Mock()
        namespace = dict(defaultdict=defaultdict, re=re, Any=Any, wandb=self.wandb)

        exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), namespace)

        self.Logger = namespace["WandbLogger"]

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


if __name__ == "__main__":
    unittest.main()
