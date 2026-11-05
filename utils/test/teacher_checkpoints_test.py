import ast
import json
import shutil
from datetime import datetime
from contextlib import nullcontext
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from omegaconf import OmegaConf

from utils.config_snapshot import register_config_resolvers, resolve_training_output_dir
from utils.teacher_checkpoints import resolve_teacher_checkpoint_groups, save_teacher_selection


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
            self.assertEqual(resolve_teacher_checkpoint_groups(str(folder), 2), [(selected, "individual", None)])
            self.assertEqual(Path(resolve_training_output_dir("checkpoint", "new", str(folder), 2)), self.root / "run")

    def test_direct_sources_without_start_iteration(self):
        save_teacher_selection(self.paths)

        for folder in (self.directory, self.directory.parent):
            self.assertEqual(resolve_teacher_checkpoint_groups(folder), [(self.paths, "individual", None)])

        self.assertEqual(resolve_teacher_checkpoint_groups(self.paths), [(self.paths, "individual", None)])
        self.assertEqual(resolve_teacher_checkpoint_groups(self.paths[0]), [(self.paths[:1], "individual", None)])

        with self.assertRaisesRegex(ValueError, "start_iteration is required"):
            resolve_teacher_checkpoint_groups(self.root / "run")

    def test_explicit_list_and_single_checkpoint(self):
        values = OmegaConf.create([str(path) for path in self.paths])

        self.assertEqual(resolve_teacher_checkpoint_groups(values, 2), [(self.paths, "individual", None)])
        self.assertEqual(resolve_teacher_checkpoint_groups(str(self.paths[0]), 2), [(self.paths[:1], "individual", None)])

        register_config_resolvers()

        cfg = OmegaConf.create({"paths": list(values), "out": "${training_output_dir:checkpoint,new,${paths},2}"})

        self.assertEqual(Path(cfg.out), self.root / "run")

    def test_training_selection_writes_manifest(self):
        source = Path(__file__).resolve().parents[2] / "training/noisy_student.py"
        node = next(node for node in ast.parse(source.read_text()).body if isinstance(node, ast.FunctionDef) and node.name == "get_checkpoint_paths")
        namespace = dict(save_teacher_selection=save_teacher_selection, resolve_teacher_checkpoint_groups=resolve_teacher_checkpoint_groups)

        exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), namespace)

        result = SimpleNamespace(top_checkpoints=self.paths, swa_checkpoints=self.paths[:1])
        cfg = SimpleNamespace(train=SimpleNamespace(teacher=self.teacher, checkpoint=SimpleNamespace(save_top_k=2, swa_save_k=1)))
        selected = namespace["get_checkpoint_paths"](cfg, result)

        self.assertEqual(resolve_teacher_checkpoint_groups(str(self.directory), 2), [(selected, "individual", None)])

    def test_legacy_folder_requires_explicit_files_even_with_averages(self):
        averages = [self.directory / name for name in ("top_k_average.pt", "swa_average.pt")]

        for path in averages:
            path.touch()

        for mode in ("individual", "average"):
            self.teacher.top_k_mode = self.teacher.swa_mode = mode

            with self.assertRaisesRegex(ValueError, "explicit checkpoint list"):
                resolve_teacher_checkpoint_groups(str(self.directory), 2)

        self.assertEqual(resolve_teacher_checkpoint_groups(averages, 2), [(averages, "individual", None)])
        self.assertEqual(resolve_teacher_checkpoint_groups(str(averages[0]), 2), [(averages[:1], "individual", None)])

    def test_missing_manifest_entry_and_mixed_runs(self):
        save_teacher_selection([self.directory / "missing.pt"])

        with self.assertRaises(FileNotFoundError):
            resolve_teacher_checkpoint_groups(str(self.directory), 2)

        other = self.root / "other/iteration/1/checkpoints/model.pt"

        other.parent.mkdir(parents=True)

        other.touch()

        with self.assertRaisesRegex(ValueError, "share one run"):
            resolve_training_output_dir("checkpoint", "new", [str(self.paths[0]), str(other)], 2)

        self.assertEqual(resolve_training_output_dir(None, "new", [str(self.paths[0]), str(other)], 2), "new")

    def test_named_checkpoint_sets_and_default_best_are_independent(self):
        save_teacher_selection(self.paths[::-1], "top_k_checkpoints.json")
        save_teacher_selection(self.paths, "swa_checkpoints.json")
        save_teacher_selection(self.paths[:1])
        counts = SimpleNamespace(save_top_k=2, swa_save_k=2)
        self.assertEqual(resolve_teacher_checkpoint_groups(self.directory, 2), [(self.paths[:1], "individual", None)])
        expected = [(self.paths[::-1], "individual", "top_k"), (self.paths, "individual", "swa")]
        self.assertEqual(resolve_teacher_checkpoint_groups(self.directory, 2, self.teacher, counts), expected)

        moved = self.root / "moved_baseline/checkpoints"
        shutil.copytree(self.directory, moved)
        expected = [([moved / path.name for path in paths], mode, name) for paths, mode, name in expected]
        self.assertEqual(resolve_teacher_checkpoint_groups(moved, 1, self.teacher, counts), expected)

    def test_resume_uses_exact_manifest_without_consulting_catalogs(self):
        average = self.directory / "top_k_average.pt"
        average.write_bytes(b"saved average")
        selected = [average, self.paths[1], average]
        save_teacher_selection(selected)
        counts = SimpleNamespace(save_top_k=1, swa_save_k=1)
        self.teacher.top_k_mode = self.teacher.swa_mode = "average"

        for invalid_catalogs in (False, True):
            if invalid_catalogs:
                for name in ("top_k", "swa"):
                    (self.directory / f"{name}_checkpoints.json").write_text("invalid catalog")
            for folder in (self.directory, self.directory.parent, self.root / "run"):
                self.assertEqual(resolve_teacher_checkpoint_groups(folder, 2, self.teacher, counts, resume=True),
                                 [(selected, "individual", None)])

        average.unlink()
        with self.assertRaisesRegex(FileNotFoundError, "Teacher checkpoint not found"):
            resolve_teacher_checkpoint_groups(self.directory, 2, self.teacher, counts, resume=True)

    def test_resume_requires_valid_manifest_even_when_catalogs_exist(self):
        save_teacher_selection(self.paths, "top_k_checkpoints.json")
        save_teacher_selection(self.paths, "swa_checkpoints.json")
        counts = SimpleNamespace(save_top_k=1, swa_save_k=1)
        manifest = self.directory / "teacher_checkpoints.json"
        with self.assertRaisesRegex(ValueError, "no teacher_checkpoints.json"):
            resolve_teacher_checkpoint_groups(self.directory, 2, self.teacher, counts, resume=True)
        for contents in ("invalid json", "[]", '["../outside.pt"]'):
            manifest.write_text(contents)
            with self.assertRaises(ValueError):
                resolve_teacher_checkpoint_groups(self.directory, 2, self.teacher, counts, resume=True)

    def test_folder_lists_mixed_sources_and_json_inputs_are_rejected(self):
        other = self.root / "other/checkpoints"
        other.mkdir(parents=True)
        manifest = self.directory / "top_k_checkpoints.json"
        save_teacher_selection(self.paths, manifest.name)
        invalid = [
            [self.directory], [self.directory, other], [self.directory, self.directory],
            [self.directory, self.paths[0]], [self.paths[0], self.directory],
            manifest, [manifest], [self.paths[0], manifest],
            OmegaConf.create([str(self.directory), str(other)]),
            OmegaConf.create([str(self.paths[0]), str(self.directory)]),
        ]
        for source in invalid:
            with self.subTest(source=source), self.assertRaisesRegex(ValueError, "teacher_checkpoint"):
                resolve_teacher_checkpoint_groups(source, 2)
        with self.assertRaisesRegex(ValueError, "teacher_checkpoint"):
            resolve_training_output_dir("checkpoint", "new", [str(self.directory), str(other)], 2)

    def test_explicit_sources_reject_filename_collisions_before_file_validation(self):
        other = self.root / "other" / self.paths[0].name
        other.parent.mkdir()
        for exists in (False, True):
            if exists:
                other.touch()
            for sources in ([self.paths[0], other], OmegaConf.create([str(self.paths[0]), str(other)])):
                with self.subTest(exists=exists, sources=sources):
                    with self.assertRaisesRegex(ValueError, "teacher_checkpoint files must have distinct filenames"):
                        resolve_teacher_checkpoint_groups(sources, 1)

    def test_json_lists_reject_invalid_missing_and_escaping_entries(self):
        manifest = self.directory / "top_k_checkpoints.json"
        for value, error in (([], ValueError), ({"top": []}, ValueError), ([42], ValueError), (["missing.pt"], FileNotFoundError), (["../outside.pt"], ValueError)):
            with self.subTest(value=value):
                manifest.write_text(json.dumps(value))
                with self.assertRaises(error):
                    resolve_teacher_checkpoint_groups(self.directory, 2, SimpleNamespace(use_top_k=True, use_swa=False), SimpleNamespace(save_top_k=1))

    def test_supervised_exports_retained_top_and_swa_sets(self):
        import torch
        from training.checkpoint import CheckpointManager

        model = torch.nn.Linear(1, 1, bias=False)
        manager = CheckpointManager(self.directory, "metrics_val/dice", "max", 3, False, "test_{epoch:03d}_{val_loss:.3f}", swa_save_k=3)
        for epoch, score in enumerate((.2, .8, .4, .3, .9, .1)):
            with torch.no_grad():
                model.weight.fill_(epoch)
            manager.step(epoch, {"metrics_val/dice": score}, model)
        manager.clear()
        result = SimpleNamespace(best_checkpoint=manager.best_model_path, top_checkpoints=manager.top_k_model_paths, swa_checkpoints=manager.swa_model_paths)
        cfg = OmegaConf.create({"train": {"pseudo_label_version": None, "pseudo_data_dir": None, "aug": {}}, "path": {"output_dir": str(self.directory.parent)}})
        source = Path(__file__).resolve().parents[2] / "training/supervised.py"
        node = next(node for node in ast.parse(source.read_text()).body if isinstance(node, ast.FunctionDef) and node.name == "train_supervised")
        namespace = dict(Path=Path, fit=Mock(return_value=result), validate_training_data=Mock(), save_teacher_selection=save_teacher_selection)
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), namespace)
        self.assertIs(namespace["train_supervised"](cfg), result)

        def epochs(manifest):
            paths = [manifest.parent / name for name in json.loads(manifest.read_text())]
            return [torch.load(path, map_location="cpu", weights_only=False)["epoch"] for path in paths]

        self.assertFalse((self.directory / "teacher_checkpoints.json").exists())
        best = resolve_teacher_checkpoint_groups(self.directory, 2, SimpleNamespace(use_top_k=True, use_swa=False), SimpleNamespace(save_top_k=1))
        self.assertEqual(best, [([manager.best_model_path], "individual", "top_k")])
        self.assertEqual(epochs(self.directory / "top_k_checkpoints.json"), [4, 1, 2])
        self.assertEqual(epochs(self.directory / "swa_checkpoints.json"), [2, 3, 4])
        self.assertFalse(any(self.directory.glob("test_005_*.pt")))

    def test_early_best_uses_available_swa_predecessors(self):
        import torch
        from training.checkpoint import CheckpointManager

        source = Path(__file__).resolve().parents[2] / "training/noisy_student.py"
        node = next(node for node in ast.parse(source.read_text()).body if isinstance(node, ast.FunctionDef) and node.name == "get_checkpoint_paths")
        average = Mock(side_effect=lambda cfg, paths, output_path, **kwargs: output_path.write_bytes(b"averaged"))
        namespace = dict(save_teacher_selection=save_teacher_selection, resolve_teacher_checkpoint_groups=resolve_teacher_checkpoint_groups, save_average_checkpoint=average)
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), namespace)
        for best_epoch in (0, 3):
            directory = self.root / f"early_best_{best_epoch}"
            model = torch.nn.Linear(1, 1, bias=False)
            manager = CheckpointManager(directory, "dice", "max", 1, False, "epoch_{epoch:03d}_{val_loss:.3f}", swa_save_k=5)
            for epoch in range(6):
                manager.step(epoch, {"dice": 1. if epoch == best_epoch else .1}, model)
            manager.clear()
            expected = manager.swa_model_paths
            save_teacher_selection(expected, "swa_checkpoints.json")
            self.assertEqual([torch.load(path, weights_only=False)["epoch"] for path in expected], list(range(best_epoch + 1)))
            result = SimpleNamespace(top_checkpoints=manager.top_k_model_paths, swa_checkpoints=expected)
            cfg = SimpleNamespace(train=SimpleNamespace(
                checkpoint=SimpleNamespace(save_top_k=1, swa_save_k=5),
                teacher=SimpleNamespace(use_top_k=False, use_swa=True, swa_mode="individual")))
            selected = namespace["get_checkpoint_paths"](cfg, result)
            self.assertEqual(selected, expected)
            cfg.train.teacher.swa_mode = "average"
            selected = namespace["get_checkpoint_paths"](cfg, result)
            self.assertEqual(selected, [directory / "swa_average.pt"])
            self.assertEqual(average.call_args.args[1], expected)
            expected[0].unlink()
            with self.assertRaises(FileNotFoundError):
                resolve_teacher_checkpoint_groups(directory, 1, cfg.train.teacher, cfg.train.checkpoint)

    def test_baseline_folder_selects_configured_top_and_swa_counts(self):
        paths = [self.directory / f"epoch_{i}.pt" for i in range(5)]
        for path in paths:
            path.touch()
        top, swa = [paths[i] for i in (4, 2, 1, 3, 0)], paths
        save_teacher_selection(top, "top_k_checkpoints.json")
        save_teacher_selection(swa, "swa_checkpoints.json")
        save_teacher_selection(top[:1])
        counts = SimpleNamespace(save_top_k=2, swa_save_k=3)
        for use_top, use_swa in ((True, False), (False, True), (True, True)):
            for top_mode in ("individual", "average"):
                for swa_mode in ("individual", "average"):
                    with self.subTest(use_top=use_top, use_swa=use_swa, top_mode=top_mode, swa_mode=swa_mode):
                        teacher = SimpleNamespace(use_top_k=use_top, use_swa=use_swa, top_k_mode=top_mode, swa_mode=swa_mode)
                        expected = ([(top[:2], top_mode, "top_k")] if use_top else []) + ([(swa[-3:], swa_mode, "swa")] if use_swa else [])
                        self.assertEqual(resolve_teacher_checkpoint_groups(self.directory.parent, 2, teacher, counts), expected)
                        self.assertEqual(resolve_teacher_checkpoint_groups(self.directory, 2, teacher, counts), expected)
        counts.save_top_k = -1
        self.teacher.use_swa = False
        (self.directory / "checkpoint_potential_maximum_k.json").write_text(json.dumps({"top_k": len(top)}))
        self.assertEqual(resolve_teacher_checkpoint_groups(self.directory, 2, self.teacher, counts), [(top, "individual", "top_k")])

    def test_all_top_checkpoints_reject_missing_or_incomplete_availability(self):
        save_teacher_selection(self.paths, "top_k_checkpoints.json")
        self.teacher.use_swa = False
        counts = SimpleNamespace(save_top_k=-1)
        metadata = self.directory / "checkpoint_potential_maximum_k.json"

        with self.assertRaisesRegex(ValueError, "top_k=-1.*complete Top-k catalog"):
            resolve_teacher_checkpoint_groups(self.directory, 2, self.teacher, counts)

        for available in (None, 1, 50, True, "2"):
            with self.subTest(available=available):
                metadata.write_text(json.dumps({"top_k": available}))
                with self.assertRaisesRegex(ValueError, "top_k=-1.*complete Top-k catalog"):
                    resolve_teacher_checkpoint_groups(self.directory, 2, self.teacher, counts)

        # A finite request can still reuse a retained subset of the source run.
        counts.save_top_k = 1
        self.assertEqual(resolve_teacher_checkpoint_groups(self.directory, 2, self.teacher, counts),
                         [(self.paths[:1], "individual", "top_k")])

    def test_all_top_checkpoints_match_actual_training_availability(self):
        import torch
        from training.checkpoint import CheckpointManager

        model = torch.nn.Linear(1, 1, bias=False)
        self.teacher.use_swa = False
        counts = SimpleNamespace(save_top_k=-1)
        for retained in (-1, 2, 10):
            with self.subTest(retained=retained):
                directory = self.root / f"retain_{retained}"
                manager = CheckpointManager(directory, "dice", "max", retained, False,
                                            "epoch_{epoch:03d}", every_n_epochs=2)
                for epoch in range(8):
                    manager.step(epoch, {"dice": epoch}, model)
                manager.clear()
                save_teacher_selection(manager.top_k_model_paths, "top_k_checkpoints.json")

                if retained == 2:
                    with self.assertRaisesRegex(ValueError, "top_k=-1.*complete Top-k catalog"):
                        resolve_teacher_checkpoint_groups(directory, 1, self.teacher, counts)
                else:
                    self.assertEqual(resolve_teacher_checkpoint_groups(directory, 1, self.teacher, counts),
                                     [(manager.top_k_model_paths, "individual", "top_k")])

    def test_short_catalogs_require_early_checkpoint_availability(self):
        import torch
        from training.checkpoint import CheckpointManager

        model = torch.nn.Linear(1, 1, bias=False)
        for group in ("top_k", "swa"):
            for epochs, interval, retained, requested in ((2, 1, 5, 5), (4, 2, 5, 5), (20, 1, 2, 5), (20, 2, 2, 5)):
                with self.subTest(group=group, epochs=epochs, interval=interval):
                    directory = self.root / f"{group}_{epochs}_{interval}"
                    manager = CheckpointManager(directory, "dice", "max", retained, False,
                                                "epoch_{epoch:03d}", every_n_epochs=interval, swa_save_k=retained)
                    for epoch in range(epochs):
                        # Top-k must consider the whole run, even with an early best.
                        score = -epoch if group == "top_k" else epoch
                        manager.step(epoch, {"dice": score}, model)
                    manager.clear()
                    paths = manager.top_k_model_paths if group == "top_k" else manager.swa_model_paths
                    save_teacher_selection(paths, f"{group}_checkpoints.json")
                    teacher = SimpleNamespace(use_top_k=group == "top_k", use_swa=group == "swa")
                    counts = SimpleNamespace(save_top_k=requested, swa_save_k=requested)
                    availability = json.loads((directory / "checkpoint_potential_maximum_k.json").read_text())
                    self.assertEqual(availability["top_k"], epochs // interval)

                    if epochs // interval < requested:
                        selected = resolve_teacher_checkpoint_groups(directory, 1, teacher, counts)
                        self.assertEqual(selected, [(paths, "individual", group)])
                    else:
                        with self.assertRaisesRegex(ValueError, f"Requested {group}={requested}"):
                            resolve_teacher_checkpoint_groups(directory, 1, teacher, counts)

    def test_incomplete_early_catalogs_are_rejected(self):
        for group in ("top_k", "swa"):
            with self.subTest(group=group):
                save_teacher_selection(self.paths, f"{group}_checkpoints.json")
                teacher = SimpleNamespace(use_top_k=group == "top_k", use_swa=group == "swa")
                counts = SimpleNamespace(save_top_k=5, swa_save_k=5)
                metadata = self.directory / "checkpoint_potential_maximum_k.json"
                for available in (None, 1, 3, True, "2"):
                    metadata.write_text(json.dumps({group: available}))
                    with self.assertRaisesRegex(ValueError, f"Requested {group}=5"):
                        resolve_teacher_checkpoint_groups(self.directory, 1, teacher, counts)

    def test_short_training_selects_and_imports_available_teachers(self):
        import torch
        from training.checkpoint import CheckpointManager

        source = Path(__file__).resolve().parents[2] / "training/noisy_student.py"
        names = {"get_checkpoint_paths", "prepare_teacher_checkpoints"}
        nodes = [node for node in ast.parse(source.read_text()).body if isinstance(node, ast.FunctionDef) and node.name in names]
        namespace = dict(Path=Path, shutil=shutil, save_teacher_selection=save_teacher_selection,
                         resolve_teacher_checkpoint_groups=resolve_teacher_checkpoint_groups)
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), namespace)
        manager = CheckpointManager(self.directory, "dice", "max", 5, False, "epoch_{epoch:03d}", swa_save_k=5)
        model = torch.nn.Linear(1, 1, bias=False)
        for epoch in range(2):
            manager.step(epoch, {"dice": epoch}, model)
        manager.clear()
        cfg = SimpleNamespace(train=SimpleNamespace(teacher=self.teacher, checkpoint=SimpleNamespace(save_top_k=5, swa_save_k=5)))
        result = SimpleNamespace(top_checkpoints=manager.top_k_model_paths, swa_checkpoints=manager.swa_model_paths)
        save_teacher_selection(result.top_checkpoints, "top_k_checkpoints.json")
        save_teacher_selection(result.swa_checkpoints, "swa_checkpoints.json")
        selected = namespace["get_checkpoint_paths"](cfg, result)
        self.assertEqual(selected, result.top_checkpoints + result.swa_checkpoints)

        groups = resolve_teacher_checkpoint_groups(self.directory, 1, self.teacher, cfg.train.checkpoint)
        output = self.root / "imported"
        imported = namespace["prepare_teacher_checkpoints"](cfg, groups, output)
        self.assertEqual([path.name for path in imported], [path.name for path in selected])
        self.assertEqual((output / "checkpoints/checkpoint_potential_maximum_k.json").read_bytes(),
                         (self.directory / "checkpoint_potential_maximum_k.json").read_bytes())
        regrouped = resolve_teacher_checkpoint_groups(output, 1, self.teacher, cfg.train.checkpoint)
        self.assertEqual([path for paths, _, _ in regrouped for path in paths], imported)

    def test_invalid_baseline_selection_fails_before_training(self):
        save_teacher_selection(self.paths, "top_k_checkpoints.json")
        save_teacher_selection(self.paths, "swa_checkpoints.json")
        counts = SimpleNamespace(save_top_k=1, swa_save_k=1)
        self.teacher.use_top_k = self.teacher.use_swa = False
        with self.assertRaisesRegex(ValueError, "enable Top-k"):
            resolve_teacher_checkpoint_groups(self.directory, 2, self.teacher, counts)
        self.teacher.use_top_k = True
        counts.save_top_k = 0
        with self.assertRaisesRegex(ValueError, "save_top_k"):
            resolve_teacher_checkpoint_groups(self.directory, 2, self.teacher, counts)
        counts.save_top_k = 1
        self.teacher.top_k_mode = "invalid"
        with self.assertRaisesRegex(ValueError, "top_k_mode"):
            resolve_teacher_checkpoint_groups(self.directory, 2, self.teacher, counts)
        self.teacher.use_top_k, self.teacher.use_swa = False, True
        counts.swa_save_k = -1
        with self.assertRaisesRegex(ValueError, "swa_save_k"):
            resolve_teacher_checkpoint_groups(self.directory, 2, self.teacher, counts)

    def test_config_selection_requires_catalogs_and_ignores_actual_teacher_log(self):
        save_teacher_selection(self.paths, "top_k_checkpoints.json")
        save_teacher_selection(self.paths[::-1], "swa_checkpoints.json")
        counts = SimpleNamespace(save_top_k=1, swa_save_k=1)
        actual = self.directory / "teacher_checkpoints.json"
        for contents in (None, "invalid audit log"):
            if contents is not None:
                actual.write_text(contents)
            expected = [(self.paths[:1], "individual", "top_k"), (self.paths[:1], "individual", "swa")]
            self.assertEqual(resolve_teacher_checkpoint_groups(self.directory, 2, self.teacher, counts), expected)
        actual.unlink()
        counts.save_top_k = 3
        with self.assertRaisesRegex(ValueError, "Requested top_k=3, but only 2"):
            resolve_teacher_checkpoint_groups(self.directory, 2, self.teacher, counts)
        counts.save_top_k, counts.swa_save_k = 1, 3
        with self.assertRaisesRegex(ValueError, "Requested swa=3, but only 2"):
            resolve_teacher_checkpoint_groups(self.directory, 2, self.teacher, counts)
        counts.swa_save_k = 1
        self.paths[0].unlink()
        with self.assertRaises(FileNotFoundError):
            resolve_teacher_checkpoint_groups(self.directory, 2, self.teacher, counts)
        (self.directory / "top_k_checkpoints.json").unlink()
        (self.directory / "swa_checkpoints.json").unlink()
        save_teacher_selection(self.paths[1:])
        with self.assertRaisesRegex(ValueError, "top_k_checkpoints.json"):
            resolve_teacher_checkpoint_groups(self.directory, 2, self.teacher, counts)

    def test_explicit_files_still_define_their_selection(self):
        selected = [*self.paths, self.paths[0]]
        save_teacher_selection(selected)
        counts = SimpleNamespace(save_top_k=1, swa_save_k=1)
        self.teacher.top_k_mode = self.teacher.swa_mode = "average"
        other = self.root / "other/explicit.pth"
        other.parent.mkdir()
        other.touch()
        selected = [self.paths[1], other, self.paths[0], self.paths[1]]
        self.assertEqual(resolve_teacher_checkpoint_groups(selected, 2, self.teacher, counts), [(selected, "individual", None)])
        self.assertEqual(resolve_teacher_checkpoint_groups(OmegaConf.create([str(path) for path in selected]), 2, self.teacher, counts), [(selected, "individual", None)])
        self.assertEqual(resolve_teacher_checkpoint_groups(self.paths[0], 2, self.teacher, counts), [(self.paths[:1], "individual", None)])

    def test_both_groups_are_combined_and_resumed_without_reaveraging(self):
        import torch
        from torch.optim.swa_utils import AveragedModel
        from training.precision import parse_precision
        from utils.device import get_device

        paths = [*self.paths, self.directory / "third.pt"]
        for path, weight in zip(paths, (0., 2., 10.)):
            torch.save({"state_dict": {"weight": torch.tensor([[weight]])}}, path)
        original = [path.read_bytes() for path in paths]
        save_teacher_selection(paths[::-1], "top_k_checkpoints.json")
        save_teacher_selection([paths[1], paths[0], paths[2]], "swa_checkpoints.json")

        def load_model_from_checkpoint(cfg, checkpoint_path, model_dtype, device):
            self.assertEqual(device, torch.device("cpu"))
            self.assertIsNone(model_dtype)
            model = torch.nn.Linear(1, 1, bias=False).to(device=device, dtype=model_dtype)
            model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=False)["state_dict"])
            return model

        source = Path(__file__).resolve().parents[2] / "training/noisy_student.py"
        nodes = [node for node in ast.parse(source.read_text()).body if isinstance(node, ast.FunctionDef) and node.name in {"prepare_teacher_checkpoints", "save_average_checkpoint", "backup_iteration_directories"}]
        namespace = dict(Path=Path, torch=torch, shutil=shutil, datetime=datetime, AveragedModel=AveragedModel, OmegaConf=OmegaConf, get_device=get_device, parse_precision=parse_precision, load_model_from_checkpoint=load_model_from_checkpoint, save_teacher_selection=save_teacher_selection)
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), namespace)
        cfg = OmegaConf.create({"model": {"name": "test"}, "train": {"precision": "32-true", "accelerator": "cpu", "checkpoint": {"save_top_k": 2, "swa_save_k": 2}, "teacher": {"use_top_k": True, "use_swa": True, "top_k_mode": "average", "swa_mode": "average"}}})

        def weights(selected):
            return [torch.load(path, map_location="cpu", weights_only=False)["state_dict"]["weight"].item() for path in selected]

        for top_mode in ("individual", "average"):
            for swa_mode in ("individual", "average"):
                with self.subTest(top_mode=top_mode, swa_mode=swa_mode):
                    cfg.train.teacher.top_k_mode, cfg.train.teacher.swa_mode = top_mode, swa_mode
                    groups = resolve_teacher_checkpoint_groups(self.directory, 2, cfg.train.teacher, cfg.train.checkpoint)
                    run = self.root / f"new_{top_mode}_{swa_mode}"
                    output = run / "iteration/0"
                    expected = ([6.] if top_mode == "average" else [10., 2.]) + ([5.] if swa_mode == "average" else [0., 10.])
                    prepared = namespace["prepare_teacher_checkpoints"](cfg, groups, output)
                    for path in paths:
                        target = output / "checkpoints" / path.name
                        self.assertTrue(target.is_file())
                        self.assertFalse(target.samefile(path))
                    self.assertEqual(weights(prepared), expected)
                    self.assertEqual([path.read_bytes() for path in paths], original)

                    # Resume restores the saved ensemble, including ready-made averages.
                    restored_groups = resolve_teacher_checkpoint_groups(run, 1, cfg.train.teacher, cfg.train.checkpoint, resume=True)
                    self.assertEqual(restored_groups, [(prepared, "individual", None)])
                    restored_groups = namespace["backup_iteration_directories"](run, self.root / "cache", 0, restored_groups)
                    average_writer = Mock(wraps=namespace["save_average_checkpoint"])
                    with patch.dict(namespace, save_average_checkpoint=average_writer):
                        restored = namespace["prepare_teacher_checkpoints"](cfg, restored_groups, output, resume=True)
                    average_writer.assert_not_called()
                    self.assertEqual(weights(restored), expected)
                    self.assertEqual(resolve_teacher_checkpoint_groups(run, 1), [(restored, "individual", None)])
                    # Raw catalogs remain usable as the source of a new ablation.
                    regrouped = resolve_teacher_checkpoint_groups(run, 1, cfg.train.teacher, cfg.train.checkpoint)
                    for (raw, mode, name), (original_raw, original_mode, original_name) in zip(regrouped, groups):
                        self.assertEqual((mode, name), (original_mode, original_name))
                        self.assertEqual(raw, [output / "checkpoints" / path.name for path in original_raw])
                        self.assertTrue(all(path.is_file() for path in raw))
                    for name, mode in (("top_k", top_mode), ("swa", swa_mode)):
                        if mode == "average":
                            average = output / "checkpoints" / f"{name}_average.pt"
                            average.unlink()
                            with self.assertRaisesRegex(FileNotFoundError, "Teacher checkpoint not found"):
                                resolve_teacher_checkpoint_groups(run, 1, cfg.train.teacher, cfg.train.checkpoint, resume=True)
                            break
                    if top_mode == swa_mode == "individual":
                        self.assertEqual(restored[0], restored[-1])  # Preserve overlap weighting.

    def test_explicit_teacher_import_can_be_resumed_repeatedly(self):
        source = Path(__file__).resolve().parents[2] / "training/noisy_student.py"
        names = {"prepare_teacher_checkpoints", "backup_iteration_directories"}
        nodes = [node for node in ast.parse(source.read_text()).body if isinstance(node, ast.FunctionDef) and node.name in names]
        average = Mock()
        namespace = dict(Path=Path, shutil=shutil, datetime=datetime, save_teacher_selection=save_teacher_selection,
                         save_average_checkpoint=average)
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), namespace)
        selected = [self.paths[1], self.paths[0], self.paths[1]]
        for index, path in enumerate(self.paths):
            path.write_bytes(f"explicit teacher {index}".encode())
        expected = [path.read_bytes() for path in selected]
        counts = SimpleNamespace(save_top_k=1, swa_save_k=1)
        self.teacher.top_k_mode = self.teacher.swa_mode = "average"
        run = self.root / "explicit_run"
        output = run / "iteration/0"
        groups = resolve_teacher_checkpoint_groups(selected, 1, self.teacher, counts)
        namespace["prepare_teacher_checkpoints"](None, groups, output)

        for _ in range(2):
            groups = resolve_teacher_checkpoint_groups(run, 1, self.teacher, counts, resume=True)
            groups = namespace["backup_iteration_directories"](run, self.root / "cache", 0, groups)
            restored = namespace["prepare_teacher_checkpoints"](None, groups, output, resume=True)
            self.assertEqual([path.read_bytes() for path in restored], expected)
            self.assertEqual(restored[0], restored[-1])
            self.assertTrue(all(path.is_relative_to(output) for path in restored))
            self.assertFalse((output / "checkpoints/top_k_checkpoints.json").exists())
            self.assertFalse((output / "checkpoints/swa_checkpoints.json").exists())
        average.assert_not_called()

    def test_average_uses_configured_device_and_precision_for_batchnorm(self):
        import torch
        from torch.optim.swa_utils import AveragedModel
        from training.precision import parse_precision
        from utils.device import get_device
        from training.batchnorm import update_batchnorm_statistics

        def model():
            return torch.nn.Sequential(torch.nn.Linear(2, 2, bias=False), torch.nn.BatchNorm1d(2))

        for path, scale in zip(self.paths, (1., 3.)):
            network = model()
            with torch.no_grad():
                network[0].weight.copy_(torch.eye(2) * scale)
                network[1].running_mean.fill_(100.)
            torch.save({"state_dict": network.state_dict()}, path)

        def load(cfg, checkpoint_path, model_dtype, device):
            self.assertEqual(device, torch.device("cpu"))
            self.assertIsNone(model_dtype)
            network = model().to(device=device, dtype=model_dtype)
            network.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=False)["state_dict"])
            return network.eval()

        batches = [{"tile": torch.tensor([[1., 2.], [3., 4.]])}, {"tile": torch.tensor([[5., 6.], [7., 8.]])}]
        source = Path(__file__).resolve().parents[2] / "training/noisy_student.py"
        node = next(node for node in ast.parse(source.read_text()).body if isinstance(node, ast.FunctionDef) and node.name == "save_average_checkpoint")
        device_selection = Mock(wraps=get_device)
        precision_selection = Mock(wraps=parse_precision)
        calibration = Mock(wraps=update_batchnorm_statistics)
        namespace = dict(torch=torch, Path=Path, AveragedModel=AveragedModel, OmegaConf=OmegaConf, load_model_from_checkpoint=load,
                         get_device=device_selection, parse_precision=precision_selection,
                         build_train_dataset=Mock(return_value=object()), build_sampler=Mock(return_value=None),
                         build_dataloader=Mock(return_value=batches), update_batchnorm_statistics=calibration)
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), namespace)
        cfg = OmegaConf.create({"model": {"name": "test"}, "train": {"accelerator": "cpu", "precision": "32-true", "use_validation_mask": False}})
        output = self.root / "configured_average.pt"
        namespace["save_average_checkpoint"](cfg, self.paths, output)
        state = torch.load(output, weights_only=False, map_location="cpu")["state_dict"]
        self.assertTrue(torch.equal(state["0.weight"], torch.eye(2) * 2))
        self.assertTrue(torch.allclose(state["1.running_mean"], torch.tensor([8., 10.])))
        self.assertTrue(all(value.device.type == "cpu" for value in state.values()))
        self.assertTrue(all(value.dtype == torch.float32 for value in state.values() if value.is_floating_point()))
        device_selection.assert_called_once_with(cfg)
        precision_selection.assert_called_once_with("32-true")
        self.assertEqual(calibration.call_args.args[2:], (torch.device("cpu"), None))

    def test_checkpoint_copies_preserve_names_and_reject_collisions(self):
        source = Path(__file__).resolve().parents[2] / "training/noisy_student.py"
        node = next(node for node in ast.parse(source.read_text()).body if isinstance(node, ast.FunctionDef) and node.name == "prepare_teacher_checkpoints")
        namespace = dict(Path=Path, shutil=shutil, save_teacher_selection=save_teacher_selection)
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), namespace)
        prepare = namespace["prepare_teacher_checkpoints"]
        first = self.paths[0]
        first.write_bytes(b"original")
        output = self.root / "copied"
        selected = prepare(None, [([first, first], "individual", None)], output)
        self.assertEqual(selected, [output / "checkpoints" / first.name] * 2)
        self.assertFalse(selected[0].samefile(first))
        selected[0].write_bytes(b"changed copy")
        self.assertEqual(first.read_bytes(), b"original")

        other = self.root / "other" / first.name
        other.parent.mkdir()
        other.write_bytes(b"different checkpoint")
        collision_output = self.root / "collision"
        with self.assertRaisesRegex(FileExistsError, "destination already exists"):
            prepare(None, [([first, other], "individual", None)], collision_output)
        self.assertEqual((collision_output / "checkpoints" / first.name).read_bytes(), b"original")
        self.assertEqual(other.read_bytes(), b"different checkpoint")

    def test_explicit_sources_only_apply_to_initial_pseudo_labels(self):
        for path, content in zip(self.paths, (b"external A", b"external B")):
            path.write_bytes(content)
        output = self.root / "new_explicit_run"
        cfg = OmegaConf.create({
            "path": {"output_dir": str(output), "tmp_dir": str(self.root / "cache")},
            "train": {"start_iteration": 1, "iterations": 3, "seed": 42,
                      "teacher_checkpoint": [str(path) for path in self.paths],
                      "checkpoint": {"save_top_k": 1, "swa_save_k": 1},
                      "teacher": {"use_top_k": True, "top_k_mode": "average", "use_swa": True, "swa_mode": "average"},
                      "student": {"aug": {}}, "predict": []},
        })
        calls, averages = [], []

        def fit(cfg, logger, output_dir, **kwargs):
            iteration = kwargs["iteration"]
            directory = output_dir / "checkpoints"
            directory.mkdir(parents=True)
            top, swa = directory / "top.pt", directory / "swa.pt"
            top.write_bytes(f"student {iteration} top".encode())
            swa.write_bytes(f"student {iteration} swa".encode())
            return SimpleNamespace(top_checkpoints=[top], swa_checkpoints=[swa])

        def average(cfg, paths, output_path=None, **kwargs):
            averages.append(paths)
            output_path.write_bytes(paths[0].read_bytes() + b" averaged")

        def generate(cfg, paths, iteration, *args, **kwargs):
            calls.append((iteration, [path.read_bytes() for path in paths]))
            if iteration == 0:
                self.assertFalse(averages)  # Explicit files bypass configured averaging.

        source = Path(__file__).resolve().parents[2] / "training/noisy_student.py"
        names = {"train_noisy_student", "prepare_teacher_checkpoints", "get_checkpoint_paths", "backup_iteration_directories"}
        nodes = [node for node in ast.parse(source.read_text()).body if isinstance(node, ast.FunctionDef) and node.name in names]
        namespace = dict(Path=Path, shutil=shutil, datetime=datetime, validate_training_data=Mock(), fit=fit,
                         resolve_teacher_checkpoint_groups=resolve_teacher_checkpoint_groups, save_teacher_selection=save_teacher_selection,
                         save_average_checkpoint=average, generate_pseudo_labels=generate, seed_everything=Mock(),
                         _get_config_for_iteration=lambda cfg, iteration: cfg, _iteration_caches=lambda cfg: nullcontext(), get_device=lambda cfg: None)
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), namespace)
        namespace["train_noisy_student"](cfg, None)
        self.assertEqual(calls, [
            (0, [b"external A", b"external B"]),
            (1, [b"student 1 top averaged", b"student 1 swa averaged"]),
            (2, [b"student 2 top averaged", b"student 2 swa averaged"]),
        ])
        self.assertEqual(len(averages), 4)
        self.assertTrue(all(path.is_relative_to(output) for paths in averages for path in paths))

    def test_backup_relocates_all_checkpoints(self):
        source = Path(__file__).resolve().parents[2] / "training/noisy_student.py"
        node = next(node for node in ast.parse(source.read_text()).body if isinstance(node, ast.FunctionDef) and node.name == "backup_iteration_directories")
        namespace = dict(Path=Path, datetime=datetime)

        exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), namespace)

        groups = [(self.paths, "average", "top_k")]
        moved_groups = namespace["backup_iteration_directories"](self.root / "run", self.root / "cache", 1, groups)
        moved, mode, name = moved_groups[0]
        self.assertEqual((mode, name), ("average", "top_k"))
        self.assertEqual(groups[0][0], self.paths)  # Caller groups are not mutated.
        self.assertTrue(all(path.is_file() for path in moved))
        self.assertTrue(all("1_backup_" in str(path) for path in moved))
        self.assertFalse(self.directory.exists())


if __name__ == "__main__":
    unittest.main()
