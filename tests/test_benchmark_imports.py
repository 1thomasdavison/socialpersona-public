from __future__ import annotations

import unittest

import user_profile_pipeline.benchmark.evaluator as evaluator


class BenchmarkImportTests(unittest.TestCase):
    def test_evaluator_import_resolves_to_scoring_package(self) -> None:
        normalized = evaluator.__file__.replace("\\", "/")
        self.assertTrue(normalized.endswith("benchmark/evaluator/__init__.py"), normalized)
        self.assertTrue(callable(evaluator.score_model_predictions_with_rubric_judge))


if __name__ == "__main__":
    unittest.main()
