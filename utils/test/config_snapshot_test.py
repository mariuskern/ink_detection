import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import warnings

from omegaconf import OmegaConf

from utils.config_snapshot import compare_configs, save_training_config, register_config_resolvers, resolve_training_output_dir, resolve_resume_run, save_wandb_identity, prepare_teacher_source


class ConfigSnapshotTests(unittest.TestCase):
    def test_continuation_rejects_different_or_new_wandb_run(self):
        with TemporaryDirectory() as folder:
            metadata = Path(folder) / ".hydra"

            metadata.mkdir()

            OmegaConf.save(OmegaConf.create({"wandb": {"run_path": "team/project/original"}}), metadata / "config.yaml")

            cfg = OmegaConf.create({"train": {"resume_output_dir": folder, "resume_run": "team/project/other"}, "path": {"output_dir": folder}})

            with self.assertRaisesRegex(ValueError, "does not match"):
                resolve_resume_run(cfg)

            cfg.train.resume_run = None

            with self.assertRaisesRegex(ValueError, "existing W&B"):
                resolve_resume_run(cfg)

            cfg.train.resume_run = "team/project/original"

            self.assertEqual(resolve_resume_run(cfg), "team/project/original")

    def test_snapshot_rejects_missing_resume_metadata_and_new_folder_collision(self):
        with TemporaryDirectory() as folder:
            cfg = OmegaConf.create({"train": {"start_iteration": 2, "resume_output_dir": folder}})
            hydra_cfg = OmegaConf.create({"runtime": {"output_dir": folder}, "overrides": {"task": []}})

            with self.assertRaisesRegex(ValueError, "old config"):
                save_training_config(cfg, hydra_cfg)

            cfg.train.resume_output_dir = "auto"

            (Path(folder) / "iteration").mkdir()

            with self.assertRaises(FileExistsError):
                save_training_config(cfg, hydra_cfg)

            self.assertFalse((Path(folder) / ".hydra").exists())

    def test_automatic_folder_resume_and_saved_identity(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            checkpoint = root / "iteration/1/checkpoints/model.pt"
            cfg = OmegaConf.create({"train": {"start_iteration": 0, "resume_run": "auto"}, "wandb": {"project": "project"}})
            hydra_cfg = OmegaConf.create({"runtime": {"output_dir": folder}, "overrides": {"task": []}})

            save_training_config(cfg, hydra_cfg)

            save_wandb_identity(cfg, hydra_cfg, SimpleNamespace(entity="team", project="project", id="actual"))

            checkpoint.parent.mkdir(parents=True)

            checkpoint.touch()

            cfg.train.start_iteration = 2
            cfg.train.teacher_checkpoint = str(root)
            cfg.train.resume_output_dir = str(root)
            cfg.path = {"output_dir": str(root)}

            self.assertEqual(resolve_resume_run(cfg), "team/project/actual")
            self.assertEqual(resolve_training_output_dir("auto", "new", str(root), 2), "new")

            for source in (str(checkpoint), [str(checkpoint)]):
                cfg.train.teacher_checkpoint = source
                cfg.train.resume_output_dir = "auto"

                self.assertIsNone(resolve_resume_run(cfg))
                self.assertEqual(resolve_training_output_dir("auto", "new", cfg.train.teacher_checkpoint, 2), "new")

                cfg.train.resume_output_dir = None

                self.assertIsNone(resolve_resume_run(cfg))
                self.assertEqual(resolve_training_output_dir(None, "new", cfg.train.teacher_checkpoint, 2), "new")

                cfg.train.resume_output_dir = "auto"

            cfg.train.teacher_checkpoint = str(root)
            cfg.train.resume_run = None

            self.assertIsNone(resolve_resume_run(cfg))

    def test_new_folder_and_explicit_wandb_override(self):
        with TemporaryDirectory() as folder:
            cfg = OmegaConf.create({"train": {"start_iteration": 1, "teacher_checkpoint": ["baseline.pt"], "resume_output_dir": None, "resume_run": "auto"}})

            self.assertIsNone(resolve_resume_run(cfg))

            cfg.train.resume_run = "team/project/source"

            with self.assertRaisesRegex(ValueError, "new W&B"):
                resolve_resume_run(cfg)

            cfg.train.resume_run = "auto"
            cfg.train.resume_output_dir = str(Path(folder) / "unused")
            cfg.path = {"output_dir": cfg.train.resume_output_dir}

            with self.assertRaisesRegex(ValueError, "old config"):
                resolve_resume_run(cfg)

    def test_teacher_inferred_from_resume_folder(self):
        with TemporaryDirectory() as folder:
            cfg = OmegaConf.create({"train": {"name": "noisy_student", "start_iteration": 2, "teacher_checkpoint": "auto", "resume_output_dir": folder}})

            prepare_teacher_source(cfg)

            self.assertEqual(cfg.train.teacher_checkpoint, folder)

            cfg.train.teacher_checkpoint = ["explicit.pt"]

            prepare_teacher_source(cfg)

            self.assertEqual(list(cfg.train.teacher_checkpoint), ["explicit.pt"])

            cfg.train.teacher_checkpoint = "auto"
            cfg.train.start_iteration = 0
            cfg.train.resume_output_dir = "auto"

            prepare_teacher_source(cfg)

            self.assertIsNone(cfg.train.teacher_checkpoint)

    def test_fresh_auto_run_without_teacher(self):
        cfg = OmegaConf.create({"train": {"start_iteration": 0, "teacher_checkpoint": None, "resume_output_dir": "auto", "resume_run": "auto"}})

        self.assertIsNone(resolve_resume_run(cfg))
        self.assertEqual(resolve_training_output_dir("auto", "new", None, 0), "new")

    def test_legacy_folder_needs_identity_or_explicit_override(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)

            (root / "iteration/1/checkpoints").mkdir(parents=True)

            (root / ".hydra").mkdir()

            cfg = OmegaConf.create({"train": {"start_iteration": 2, "teacher_checkpoint": folder, "resume_output_dir": folder, "resume_run": "auto"}})
            cfg.path = {"output_dir": folder}
            old = OmegaConf.create({"train": {"resume_run": "team/project/old", "resume_mode": "new_run"}})

            OmegaConf.save(old, root / ".hydra/config.yaml")

            with self.assertRaisesRegex(ValueError, "once"):
                resolve_resume_run(cfg)

            for mode in ("append", "rewind"):
                old.train.resume_mode = mode

                OmegaConf.save(old, root / ".hydra/config.yaml")

                with self.assertRaisesRegex(ValueError, "once"):
                    resolve_resume_run(cfg)

            cfg.train.resume_run = "team/project/old"

            self.assertEqual(resolve_resume_run(cfg), "team/project/old")

            cfg.train.resume_run = None

            with self.assertRaisesRegex(ValueError, "existing W&B"):
                resolve_resume_run(cfg)

    def test_resume_lookup_uses_effective_output_and_checkpoint_mode(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            cfg = OmegaConf.create({"train": {"resume_run": "auto", "resume_output_dir": "checkpoint"}, "path": {"output_dir": folder}})

            with self.assertRaisesRegex(ValueError, "old config"):
                resolve_resume_run(cfg)

            (root / ".hydra").mkdir()

            OmegaConf.save(OmegaConf.create({"wandb": {"run_path": "team/project/target"}}), root / ".hydra/config.yaml")

            self.assertEqual(resolve_resume_run(cfg), "team/project/target")

            # The effective destination wins over a different configured source.
            cfg.train.resume_output_dir = "some/other/run"

            self.assertEqual(resolve_resume_run(cfg), "team/project/target")

    def test_output_folder_selection(self):
        register_config_resolvers()

        for profile in ("local", "alex"):
            path = OmegaConf.load(Path(__file__).resolve().parents[2] / f"conf/path/{profile}.yaml")
            cfg = OmegaConf.create({"path": OmegaConf.to_container(path), "train": {"name": "noisy_student", "resume_output_dir": None}, "model": {"name": "model"}, "experiment": {"name": "new"}})

            with patch.dict(os.environ, OUTPUT_DIR="outputs"):
                self.assertEqual(cfg.path.output_dir, "outputs/noisy_student/model/new")

                cfg.train.resume_output_dir = "outputs/previous_run"

                self.assertEqual(cfg.path.output_dir, "outputs/previous_run")

                cfg.train.resume_output_dir = None

                self.assertEqual(cfg.path.output_dir, "outputs/noisy_student/model/new")

                cfg.train.teacher_checkpoint = "outputs/old/iteration/1/checkpoints/teacher.pt"
                cfg.train.resume_output_dir = "checkpoint"

                self.assertEqual(Path(cfg.path.output_dir), Path("outputs/old").resolve())

    def test_checkpoint_output_requires_standard_layout(self):
        for checkpoint in (None, "teacher.pt", "old/iteration/invalid/checkpoints/teacher.pt"):
            with self.assertRaises(ValueError):
                resolve_training_output_dir("checkpoint", "new", checkpoint)

    def test_new_output_folder_does_not_compare_or_copy_teacher_config(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            old_output = root / "old"
            cfg = OmegaConf.create({"train": {"start_iteration": 0}, "model": {"width": 8}})
            old_hydra = OmegaConf.create({"runtime": {"output_dir": str(old_output)}, "overrides": {"task": []}})

            save_training_config(cfg, old_hydra)

            checkpoint = old_output / "iteration/1/checkpoints/model.pt"

            checkpoint.parent.mkdir(parents=True)

            checkpoint.touch()

            before = (old_output / ".hydra/config.yaml").read_bytes()
            cfg.train.start_iteration = 2
            cfg.train.teacher_checkpoint = str(checkpoint)
            cfg.model.width = 16
            new_hydra = OmegaConf.create({"runtime": {"output_dir": str(root / "new")}, "overrides": {"task": []}})

            save_training_config(cfg, new_hydra)

            self.assertEqual((old_output / ".hydra/config.yaml").read_bytes(), before)
            self.assertEqual(list((root / "new").glob(".hydra_backup_*")), [])
            self.assertEqual(OmegaConf.load(root / "new/.hydra/config.yaml").model.width, 16)

    def test_nested_values_lists_and_missing_keys(self):
        changed, added, removed = compare_configs( {"model": {"width": 8}, "aug": [{"p": .5}], "old": True}, {"model": {"width": 16}, "aug": [{"p": .5, "new": True}], "new": 1}, )

        self.assertEqual(changed, ["model.width: 8 -> 16"])
        self.assertEqual(added, ["aug[0].new", "new"])
        self.assertEqual(removed, ["old"])
        self.assertTrue(compare_configs({"data": [1]}, {"data": [1, 2]})[0])
        self.assertTrue(compare_configs({"flag": True}, {"flag": 1})[0])

    def test_snapshots_and_failed_validation_preserve_previous_config(self):
        with TemporaryDirectory() as folder:
            hydra_cfg = OmegaConf.create({"runtime": {"output_dir": folder}, "overrides": {"task": []}})
            cfg = OmegaConf.create({"train": {"start_iteration": 0}, "model": {"width": 8}})

            save_training_config(cfg, hydra_cfg)

            directory = Path(folder) / ".hydra"
            original = (directory / "config.yaml").read_bytes()
            cfg.train.start_iteration = 2
            cfg.train.resume_output_dir = folder
            cfg.train.resume_run = "entity/project/run"
            cfg.model.new_option = True

            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")

                save_training_config(cfg, hydra_cfg)

            self.assertTrue(any("model.new_option" in str(w.message) for w in caught))

            backup = next(Path(folder).glob(".hydra_backup_*"))

            self.assertEqual((backup / "config.yaml").read_bytes(), original)

            accepted = (directory / "config.yaml").read_bytes()
            cfg.model.width = 16

            with self.assertRaisesRegex(ValueError, "model.width"):
                save_training_config(cfg, hydra_cfg)

            self.assertEqual((directory / "config.yaml").read_bytes(), accepted)
            self.assertEqual(len(list(Path(folder).glob(".hydra_backup_*/attempted_config.resolved.yaml"))), 1)

    def test_legacy_config_warns_and_preserves_hydra_metadata(self):
        with TemporaryDirectory() as folder:
            directory = Path(folder) / ".hydra"

            directory.mkdir()

            (directory / "config.yaml").write_text("train:\n  start_iteration: 0\nmodel:\n  width: 8\n")

            (directory / "overrides.yaml").write_text("- model.width=8\n")

            cfg = OmegaConf.create({"train": {"start_iteration": 2, "resume_output_dir": folder}, "model": {"width": 8}})
            hydra_cfg = OmegaConf.create({"runtime": {"output_dir": folder}, "overrides": {"task": []}})

            with self.assertWarnsRegex(UserWarning, "unresolved"):
                save_training_config(cfg, hydra_cfg)

            self.assertEqual(next(Path(folder).glob(".hydra_backup_*/overrides.yaml")).read_text(), "- model.width=8\n")

    def test_resolved_snapshot_detects_changed_environment(self):
        with TemporaryDirectory() as folder:
            hydra_cfg = OmegaConf.create({"runtime": {"output_dir": folder}, "overrides": {"task": []}})
            cfg = OmegaConf.create({"train": {"start_iteration": 0}, "path": {"data_dir": "${oc.env:SNAPSHOT_TEST_DATA}"}})

            with patch.dict(os.environ, SNAPSHOT_TEST_DATA="old_data"):
                save_training_config(cfg, hydra_cfg)

            cfg.train.start_iteration = 2
            cfg.train.resume_output_dir = folder

            with patch.dict(os.environ, SNAPSHOT_TEST_DATA="new_data"):
                with self.assertRaisesRegex(ValueError, "path.data_dir"):
                    save_training_config(cfg, hydra_cfg)

    def test_real_hydra_startup_does_not_overwrite_old_config(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)

            (root / "config.yaml").write_text( "train:\n  start_iteration: 0\nmodel:\n  width: 8\n" "hydra:\n  output_subdir: null\n  run:\n    dir: ${oc.env:SNAPSHOT_TEST_OUTPUT}\n" )

            (root / "entry.py").write_text( "import hydra\nfrom hydra.core.hydra_config import HydraConfig\n" "from utils.config_snapshot import save_training_config\n" "@hydra.main(version_base=None, config_path='.', config_name='config')\n" "def main(cfg):\n    save_training_config(cfg, HydraConfig.get())\n" "if __name__ == '__main__':\n    main()\n" )

            env = dict(os.environ, SNAPSHOT_TEST_OUTPUT=str(root / "output"))
            env["PYTHONPATH"] = os.pathsep.join(sys.path)
            command = [sys.executable, str(root / "entry.py")]
            first = subprocess.run(command, env=env, capture_output=True, text=True)

            self.assertEqual(first.returncode, 0, first.stderr)

            config = root / "output/.hydra/config.yaml"
            before = config.read_bytes()
            second = subprocess.run(command + ["train.start_iteration=2", "model.width=16", "+train.resume_output_dir=" + str(root / "output").replace("\\", "/")], env=env, capture_output=True, text=True)

            self.assertNotEqual(second.returncode, 0)
            self.assertIn("model.width", second.stderr)
            self.assertEqual(config.read_bytes(), before)
            self.assertEqual(len(list((root / "output").glob(".hydra_backup_*"))), 1)


if __name__ == "__main__":
    unittest.main()
