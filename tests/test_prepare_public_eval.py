from __future__ import annotations

import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("prepare_public_eval", ROOT / "scripts/prepare_public_eval.py")
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("Cannot import public input preparation script")
PREPARE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PREPARE)


class PreparePublicEvalTests(unittest.TestCase):
    def test_rejects_release_data_directory_and_descendants_before_writing(self):
        with tempfile.TemporaryDirectory() as tmp:
            release = Path(tmp) / "release"
            data = release / "data"
            data.mkdir(parents=True)
            marker = data / "preserve.txt"
            marker.write_text("unchanged", encoding="utf-8")
            for output in (release, data, data / "nested", data / "nested" / ".."):
                with self.subTest(output=output), patch("sys.argv", [
                    "prepare_public_eval", "--release-dir", str(release), "--output-dir", str(output),
                ]):
                    with self.assertRaisesRegex(SystemExit, "must not overwrite released data"):
                        PREPARE.main()
            self.assertEqual(marker.read_text(encoding="utf-8"), "unchanged")
            self.assertEqual(list(data.iterdir()), [marker])

    def test_prepares_public_profiles_in_separate_output_directory(self):
        from user_profile_pipeline.release_privacy import verify_data

        # The released 100-user package exercises validation and profile copying together.
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "public_inputs"
            with patch("sys.argv", [
                "prepare_public_eval", "--release-dir", str(ROOT), "--output-dir", str(output),
            ]):
                PREPARE.main()
            profiles = sorted((output / "gold").glob("*.json"))
            self.assertEqual(len(profiles), 100)
            for profile in profiles:
                source = ROOT / "data/users" / profile.stem / "gold_profile.json"
                self.assertEqual(profile.read_bytes(), source.read_bytes())
            self.assertEqual(verify_data(ROOT), [])


if __name__ == "__main__":
    unittest.main()
