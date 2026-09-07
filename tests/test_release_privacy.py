from collections import Counter
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from user_profile_pipeline.release_privacy import (
    BUCKETS, MODELS, VOCAB, build_release, extract_topics, read_posts,
    resolved_phases, source_phase_map, verify_data, write_json, write_manifest,
)


class PublicPrivacyTests(unittest.TestCase):
    def fixture(self, root):
        source = root / "source"
        for i in range(5):
            d = source / f"source_{i}"
            d.mkdir(parents=True)
            posts = [{"post_id": f"private_{j}", "created_at": f"2025-01-{8-j:02d}",
                      "text": "SECRET_PERSON lives at SECRET_ADDRESS. Cooking\u2028and coffee. @secret #reading "
                              "https://secret.example/music personal@example.com" + (" Cycling." if i == 0 else ""),
                      "media": [{"sha256": "private_media_fingerprint"}]} for j in range(8)]
            (d / "posts.jsonl").write_text("".join(json.dumps(p, ensure_ascii=False) + "\n" for p in posts), encoding="utf-8")
            domains = []
            for domain in VOCAB:
                row = {"domain": domain, "status": "active", **{b: [] for b in BUCKETS}}
                row["long_term_interests"] = [{"label": "SECRET_PERSON cooking", "evidence_post_ids": ["private_0"], "evidence_post_indices": [7]}]
                domains.append(row)
            write_json(d / "gold_profile.json", {"user_id": d.name, "domains": domains})
            write_json(d / "image_captions.json", {"captions": {"old_image": {"model_captions": {
                model: {"summary": "SECRET_PERSON at SECRET_ADDRESS with coffee"} for model in MODELS}}}})
        return source

    def test_allowlisted_derivative_and_temporal_references(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = self.fixture(root)
            out = root / "public"
            report = build_release(source, out, root / "private" / "mapping.json", expected_users=5)
            self.assertEqual(report["public_counts"]["timeline_summaries"], 20)
            self.assertEqual(verify_data(out, expected_users=5), [])
            public_text = "\n".join(p.read_text(encoding="utf-8") for p in out.rglob("*") if p.is_file())
            for forbidden in ("SECRET_PERSON", "SECRET_ADDRESS", "private_0", "private_media_fingerprint", "2025-01", "secret.example", "personal@example.com", "cycling"):
                self.assertNotIn(forbidden, public_text)
            profile = json.loads((out / "data/users/participant_001/gold_profile.json").read_text())
            self.assertEqual(profile["domains"][0]["long_term_interests"][0]["evidence_post_indices"], [3])
            row_path = out / "data/users/participant_001/posts.jsonl"
            rows = read_posts(row_path)
            rows[0]["text"] += " SECRET_PERSON"
            row_path.write_text("".join(json.dumps(row) + "\n" for row in rows))
            write_manifest(out)
            self.assertTrue(any("free text" in issue for issue in verify_data(out, expected_users=5)))

    def test_unknown_fields_and_public_mapping_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = self.fixture(root)
            out = root / "public"
            with self.assertRaises(ValueError):
                build_release(source, out, out / "mapping.json", expected_users=5)
            self.assertFalse(out.exists())
            build_release(source, out, root / "mapping.json", expected_users=5)
            path = out / "data/users/participant_001/gold_profile.json"
            value = json.loads(path.read_text())
            value["identity"] = "SECRET_PERSON"
            write_json(path, value)
            write_manifest(out)
            self.assertTrue(any("unexpected fields" in issue for issue in verify_data(out, expected_users=5)))

    def test_no_ner_or_verbatim_fallback(self):
        self.assertEqual(extract_topics("张三住在某街道，电话12345。"), set())
        self.assertEqual(extract_topics("@running #music https://example.com/coffee"), set())
        self.assertEqual(extract_topics("I enjoy cooking and coffee."), {"cooking", "coffee and tea"})

    def test_missing_ids_do_not_fall_back_to_wrong_indices(self):
        audit = Counter()
        self.assertEqual(resolved_phases({"evidence_post_ids": ["missing"], "evidence_post_indices": [0]}, {}, {0: 1}, audit), [])
        self.assertEqual(audit["unresolved_source_evidence_ids"], 1)
        with self.assertRaises(ValueError):
            source_phase_map([{"post_id": "duplicate", "created_at": "2025-01-01"}] * 4)

    def test_exact_match_smoke_never_constructs_an_api_client(self):
        from user_profile_pipeline.benchmark.profile_eval.evaluator import BenchmarkEvaluator
        from user_profile_pipeline.benchmark.profile_eval.io import _resolve_model_specs
        from user_profile_pipeline.benchmark.profile_eval.tasks import build_tasks_from_gold_exports
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = self.fixture(root)
            out = root / "public"
            build_release(source, out, root / "mapping.json", expected_users=5)
            dirs = list((out / "data/users").iterdir())
            tasks = build_tasks_from_gold_exports(gold_export_paths=[d / "gold_profile.json" for d in dirs],
                       posts_by_user={d.name: d / "posts.jsonl" for d in dirs}, max_posts=None)
            evaluator = BenchmarkEvaluator(output_dir=root / "results", cache_dir=None,
                model_specs=_resolve_model_specs(["mock_oracle"]), anchor_match_model="exact_match",
                visual_mode="wo_text_image", profile_input_mode="text_only")
            with patch("user_profile_pipeline.benchmark.profile_eval.evaluator.OpenAICompatibleChatClient", side_effect=AssertionError("API client must not be constructed")), patch("requests.Session.request", side_effect=AssertionError("Network must not be called")):
                report = evaluator.run(tasks)
            self.assertEqual(report["anchor_match_model"], "exact_match")
            evaluator.anchor_match_force_llm = True
            with self.assertRaises(ValueError):
                evaluator._build_anchor_match_client()


if __name__ == "__main__":
    unittest.main()
