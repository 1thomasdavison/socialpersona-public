from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from user_profile_pipeline.benchmark.evaluator.adapters import pred_row_to_canonical
from user_profile_pipeline.benchmark.evaluator.precheck import validate_prediction
from user_profile_pipeline.benchmark.profile_eval.evaluator import BenchmarkEvaluator
from user_profile_pipeline.benchmark.profile_eval.normalization import _normalize_prediction
from user_profile_pipeline.benchmark.profile_eval.specs import (
    PROFILE_EVAL_PROTOCOL_VERSION,
    DomainEvalTask,
    InterestAnchor,
    ModelSpec,
)


def make_task() -> DomainEvalTask:
    return DomainEvalTask(
        task_id="sample__gaming", source_tag="sample", user_id="sample",
        domain="gaming", domain_definition="Playing and following games",
        gold_status="active", gold_summary_natural="",
        gold_long_term_anchors=[InterestAnchor(f"stable theme {i}") for i in range(8)],
        gold_short_term_anchors=[InterestAnchor(f"recent theme {i}") for i in range(7)],
        gold_domain_representative_evidence_post_indices=[], posts=[], gold_path="sample.json",
    )


class ProfileExtractionTests(unittest.TestCase):
    def test_all_methods_keep_and_score_more_than_six_interests(self) -> None:
        task = make_task()
        spec = ModelSpec(name="mock_oracle", provider="mock_oracle")
        with tempfile.TemporaryDirectory() as tmp:
            for method in ("direct", "hierarchical", "extractive"):
                with self.subTest(method=method):
                    evaluator = BenchmarkEvaluator(
                        output_dir=Path(tmp) / method, cache_dir=None, model_specs=[spec],
                        profile_method=method, anchor_match_model="exact_match",
                    )
                    if method == "direct":
                        row = evaluator._predict_pending(tasks=[task], spec=spec)[task.task_id]
                    else:
                        profile = {
                            "active": True,
                            "stable_interests": [{"interest": a.label} for a in task.gold_long_term_anchors],
                            "recent_interests": [{"interest": a.label} for a in task.gold_short_term_anchors],
                        }
                        row = evaluator._method_profile_to_prediction(
                            task=task, profile=profile, final_profile={"profile": {task.domain: profile}},
                            post_id_to_index={}, model_name=spec.name, artifact_fields={},
                        )
                    self.assertEqual(len(row["pred_long_term_anchors"]), 8)
                    self.assertEqual(len(row["pred_short_term_anchors"]), 7)
                    precheck = validate_prediction(pred_row_to_canonical(row), task.posts)
                    self.assertTrue(precheck.valid, precheck.format_errors)
                    self.assertEqual(len(precheck.cleaned_prediction.interest_anchors), 15)
                    metrics = evaluator._compute_metrics(tasks=[task], predictions={task.task_id: row})
                    self.assertEqual(metrics["interest_tag_f1"], 1.0)
                    self.assertEqual(metrics["per_task_overview"][0]["anchor_match_tp"], 15)

    def test_empty_buckets_and_inactive_abstention(self) -> None:
        for status in ("active", "inactive"):
            with self.subTest(status=status):
                row = _normalize_prediction(
                    task=make_task(), model_name="test", raw_text="",
                    parsed={"status": status, "long_term_interest_tags": [],
                            "short_term_interest_tags": [f"theme {i}" for i in range(10)]},
                )
                self.assertEqual(row["pred_status"], status)
                self.assertEqual(row["pred_long_term_anchors"], [])
                self.assertEqual(len(row["pred_short_term_anchors"]), 10 if status == "active" else 0)
        row = _normalize_prediction(
            task=make_task(), model_name="test", raw_text="",
            parsed={"status": "active", "long_term_interest_tags": [], "short_term_interest_tags": []},
        )
        self.assertEqual(row["pred_status"], "active")
        self.assertTrue(validate_prediction(pred_row_to_canonical(row), []).valid)

    def test_evidence_bounds_still_reject_invalid_predictions(self) -> None:
        row = _normalize_prediction(
            task=make_task(), model_name="test", raw_text="",
            parsed={"status": "active", "long_term_interest_anchors": [
                {"label": "strategy games", "evidence_post_indices": [99]}
            ]},
        )
        self.assertTrue(validate_prediction(pred_row_to_canonical(row), []).hard_fail)

    def test_resume_requires_current_protocol_and_input_metadata(self) -> None:
        task = make_task()
        spec = ModelSpec(name="mock_oracle", provider="mock_oracle")
        with tempfile.TemporaryDirectory() as tmp:
            evaluator = BenchmarkEvaluator(
                output_dir=Path(tmp), cache_dir=None, model_specs=[spec], anchor_match_model="exact_match",
            )
            fresh = evaluator._run_model(tasks=[task], spec=spec, force=False)[task.task_id]
            self.assertEqual(fresh["profile_eval_protocol_version"], PROFILE_EVAL_PROTOCOL_VERSION)
            cache_path = Path(tmp) / "cache/text_image/mock_oracle/sample__gaming.json"
            self.assertTrue(cache_path.is_file())
            with patch.object(evaluator, "_predict_pending", wraps=evaluator._predict_pending) as predict:
                resumed = evaluator._run_model(tasks=[task], spec=spec, force=False)[task.task_id]
                self.assertTrue(resumed["from_cache"])
                predict.assert_not_called()
                for key, value in (
                    ("profile_eval_protocol_version", None),
                    ("profile_eval_protocol_version", "old_prompt"),
                    ("profile_method", "hierarchical"),
                    ("profile_input_mode", "text_only"),
                    ("visual_mode", "native"),
                ):
                    with self.subTest(key=key, value=value):
                        stale = dict(fresh)
                        if value is None:
                            stale.pop(key)
                        else:
                            stale[key] = value
                        cache_path.write_text(json.dumps(stale), encoding="utf-8")
                        predict.reset_mock()
                        regenerated = evaluator._run_model(tasks=[task], spec=spec, force=False)[task.task_id]
                        self.assertFalse(regenerated["from_cache"])
                        predict.assert_called_once()


if __name__ == "__main__":
    unittest.main()
