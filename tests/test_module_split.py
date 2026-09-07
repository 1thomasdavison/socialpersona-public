from __future__ import annotations

import unittest

from user_profile_pipeline.benchmark.profile_eval import BenchmarkEvaluator
from user_profile_pipeline.benchmark.profile_eval import model_selection
from user_profile_pipeline.personalized_dialogue.runner import build_parser, run


class ModuleSplitTests(unittest.TestCase):
    def test_profile_evaluator_methods_are_split_by_responsibility(self) -> None:
        self.assertTrue(BenchmarkEvaluator._predict_hierarchical.__module__.endswith("profile_eval.profile_prediction"))
        self.assertTrue(BenchmarkEvaluator._predict_vertex_batch.__module__.endswith("profile_eval.inference"))
        self.assertTrue(BenchmarkEvaluator._compute_metrics.__module__.endswith("profile_eval.metrics"))

    def test_profile_model_parser(self) -> None:
        self.assertEqual(model_selection._parse_models_arg("a, b c"), ["a", "b", "c"])

    def test_dialogue_entry_points_resolve_to_modular_implementation(self) -> None:
        self.assertTrue(build_parser.__module__.endswith("personalized_dialogue.runner.cli"))
        self.assertTrue(run.__module__.endswith("personalized_dialogue.runner.orchestration"))
        destinations = {action.dest for action in build_parser()._actions}
        self.assertTrue({"dialogue_root", "models", "output_dir"}.issubset(destinations))


if __name__ == "__main__":
    unittest.main()
