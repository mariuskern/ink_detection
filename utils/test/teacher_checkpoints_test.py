import ast
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

from omegaconf import OmegaConf

from utils.config_snapshot import register_config_resolvers, resolve_training_output_dir
from utils.teacher_checkpoints import resolve_teacher_checkpoints, save_teacher_selection


class TeacherCheckpointTests(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()

        self.addCleanup(self.tmp.cleanup)

        self.root = Path(self.tmp.name)
        self.directory = self.root / "run/iteration/1/checkpoints"

        self.directory.mkdir(parents=True)

        self.paths = [self.directory / name for name in ("first.pt", "second.pt")]

        for path in self.paths:
            path.touch()

        self.teacher = SimpleNamespace(use_top_k=True, top_k_mode="individual", use_swa=True, swa_mode="individual")

    def test_manifest_preserves_exact_selection_including_overlap(self):
        selected = [*self.paths, self.paths[0]]

        save_teacher_selection(selected)

        for folder in (self.directory, self.directory.parent, self.root / "run"):
            self.assertEqual(resolve_teacher_checkpoints(str(folder), 2), selected)
            self.assertEqual(Path(resolve_training_output_dir("checkpoint", "new", str(folder), 2)), self.root / "run")

    def test_explicit_list_and_single_checkpoint(self):
        values = OmegaConf.create([str(path) for path in self.paths])

        self.assertEqual(resolve_teacher_checkpoints(values, 2), self.paths)
        self.assertEqual(resolve_teacher_checkpoints(str(self.paths[0]), 2), self.paths[:1])

        register_config_resolvers()

        cfg = OmegaConf.create({"paths": list(values), "out": "${training_output_dir:checkpoint,new,${paths},2}"})

        self.assertEqual(Path(cfg.out), self.root / "run")

    def test_training_selection_writes_manifest(self):
        source = Path(__file__).resolve().parents[2] / "training/noisy_student.py"
        node = next(node for node in ast.parse(source.read_text()).body if isinstance(node, ast.FunctionDef) and node.name == "get_checkpoint_paths")
        namespace = dict(save_teacher_selection=save_teacher_selection)

        exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), namespace)

        result = SimpleNamespace(top_checkpoints=self.paths, swa_checkpoints=self.paths[:1])
        cfg = SimpleNamespace(train=SimpleNamespace(teacher=self.teacher))
        selected = namespace["get_checkpoint_paths"](cfg, result, None)

        self.assertEqual(resolve_teacher_checkpoints(str(self.directory), 2), selected)

    def test_legacy_folder_requires_explicit_files_even_with_averages(self):
        averages = [self.directory / name for name in ("top_k_average.pt", "swa_average.pt")]

        for path in averages:
            path.touch()

        for mode in ("individual", "average"):
            self.teacher.top_k_mode = self.teacher.swa_mode = mode

            with self.assertRaisesRegex(ValueError, "explicit checkpoint list"):
                resolve_teacher_checkpoints(str(self.directory), 2)

        self.assertEqual(resolve_teacher_checkpoints(averages, 2), averages)
        self.assertEqual(resolve_teacher_checkpoints(str(averages[0]), 2), averages[:1])

    def test_missing_manifest_entry_and_mixed_runs(self):
        save_teacher_selection([self.directory / "missing.pt"])

        with self.assertRaises(FileNotFoundError):
            resolve_teacher_checkpoints(str(self.directory), 2)

        other = self.root / "other/iteration/1/checkpoints/model.pt"

        other.parent.mkdir(parents=True)

        other.touch()

        with self.assertRaisesRegex(ValueError, "share one run"):
            resolve_training_output_dir("checkpoint", "new", [str(self.paths[0]), str(other)], 2)

        self.assertEqual(resolve_training_output_dir(None, "new", [str(self.paths[0]), str(other)], 2), "new")

    def test_backup_relocates_all_checkpoints(self):
        source = Path(__file__).resolve().parents[2] / "training/noisy_student.py"
        node = next(node for node in ast.parse(source.read_text()).body if isinstance(node, ast.FunctionDef) and node.name == "backup_iteration_directories")
        namespace = dict(Path=Path, datetime=datetime)

        exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), namespace)

        moved = namespace["backup_iteration_directories"](self.root / "run", self.root / "cache", 1, self.paths)

        self.assertTrue(all(path.is_file() for path in moved))
        self.assertTrue(all("1_backup_" in str(path) for path in moved))
        self.assertFalse(self.directory.exists())


if __name__ == "__main__":
    unittest.main()
