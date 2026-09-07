from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = ROOT / "scripts/run_profile_eval_single_model_jobs.py"
SPEC = importlib.util.spec_from_file_location("profile_runner", RUNNER_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Cannot import runner from {RUNNER_PATH}")
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


class ProfileRunnerTests(unittest.TestCase):
    def test_batch_endpoints_include_partial_final_batch(self) -> None:
        self.assertEqual(RUNNER.batch_endpoints(25, 10), [10, 20, 25])
        self.assertEqual(RUNNER.batch_endpoints(5, 10), [5])

    def test_invalid_batch_size_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "batch_size"):
            RUNNER.batch_endpoints(10, 0)

    def test_gold_user_limit_can_select_a_smoke_test_subset(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            gold_root = root / "gold"
            gold_root.mkdir()
            names = [f"user_{index}.json" for index in range(3)]
            for name in names:
                (gold_root / name).write_text('{"domains": []}\n', encoding="utf-8")
            list_path = root / "gold_files.txt"
            list_path.write_text("\n".join(names) + "\n", encoding="utf-8")

            selected = RUNNER.fixed_gold_files(list_path, gold_root, 2)
            self.assertEqual([path.name for path in selected], names[:2])

    def test_duplicate_gold_users_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            gold_root = root / "gold"
            gold_root.mkdir()
            (gold_root / "user.json").write_text('{"domains": []}\n', encoding="utf-8")
            list_path = root / "gold_files.txt"
            list_path.write_text("user.json\nuser.json\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "duplicate gold users"):
                RUNNER.fixed_gold_files(list_path, gold_root, 0)


if __name__ == "__main__":
    unittest.main()
