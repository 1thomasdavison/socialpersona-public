from __future__ import annotations

import unittest

from user_profile_pipeline.benchmark.profile_eval.io import _resolve_model_specs
from user_profile_pipeline.benchmark.profile_eval.modes import (
    _profile_input_mode_cache_key,
    normalize_profile_input_mode,
    normalize_profile_method,
)
from user_profile_pipeline.domain_llm.validator import DomainSummaryValidator
from user_profile_pipeline.image_text import normalize_visual_mode


class CurrentContractTests(unittest.TestCase):
    def test_historical_mode_aliases_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            normalize_profile_method("baseline")
        with self.assertRaises(ValueError):
            normalize_profile_input_mode("text-image")
        with self.assertRaises(ValueError):
            normalize_visual_mode("images")

    def test_models_must_be_registered(self) -> None:
        with self.assertRaises(ValueError):
            _resolve_model_specs(["unregistered-model"])

    def test_paper_cache_namespace_remains_resumable(self) -> None:
        self.assertEqual(
            _profile_input_mode_cache_key(
                visual_mode="text_image",
                profile_input_mode="text_image_captions_timestamps",
            ),
            "text_image",
        )

    def test_domain_interest_rows_use_one_field_set(self) -> None:
        validator = DomainSummaryValidator()
        rows = validator._clean_bucket(
            items=[
                {
                    "label": "strategy games",
                    "canonical_tags": [],
                    "all_evidence_post_ids": ["p1", "p2"],
                    "representative_evidence_post_ids": ["p1"],
                }
            ],
            domain_pack={"tag_clusters": []},
            canonical_to_evidence={},
            member_to_canonical={},
            allowed_ids={"p1", "p2"},
            bucket_name="stable_interests",
            actions=[],
        )

        self.assertEqual(rows[0]["label"], "strategy games")
        self.assertEqual(rows[0]["all_evidence_post_ids"], ["p1", "p2"])
        self.assertNotIn("interest", rows[0])
        self.assertNotIn("evidence_post_ids", rows[0])


if __name__ == "__main__":
    unittest.main()
