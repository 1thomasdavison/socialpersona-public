from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from user_profile_pipeline.config import load_domain_configs, load_yaml
from user_profile_pipeline.schemas import DOMAIN_NAMES


ROOT = Path(__file__).resolve().parents[1]


class ConfigTests(unittest.TestCase):
    def test_domain_example_covers_benchmark_taxonomy(self) -> None:
        configs = load_domain_configs(ROOT / "configs/domains.yaml.example")
        self.assertEqual([config.name for config in configs], DOMAIN_NAMES)
        self.assertIn("hiking", configs[0].seed_keywords)

    def test_empty_yaml_loads_as_empty_mapping(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "empty.yaml"
            path.write_text("", encoding="utf-8")
            self.assertEqual(load_yaml(path), {})

    def test_non_mapping_yaml_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "list.yaml"
            path.write_text("- value\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "top-level mapping"):
                load_yaml(path)

    def test_missing_config_points_to_available_example(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "model.yaml"
            path.with_name("model.yaml.example").write_text("model: test\n", encoding="utf-8")
            with self.assertRaisesRegex(FileNotFoundError, "Copy .*model.yaml.example"):
                load_yaml(path)


if __name__ == "__main__":
    unittest.main()
