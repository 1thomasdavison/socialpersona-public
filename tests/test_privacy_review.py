import importlib.util
from pathlib import Path
import unittest

SPEC = importlib.util.spec_from_file_location("privacy_review", Path(__file__).resolve().parents[1] / "scripts/review_release_privacy.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class PrivacyReviewTests(unittest.TestCase):
    def test_quarantines_placeholder_edits_and_invented_spans(self):
        batch = [{"id": "r1", "text": "[PERSON] likes coffee with Alice."}]
        result, rejected = MODULE.validate_edits({"batch_id": "0", "edits": [
            {"id": "r1", "spans": [
                {"find": "PERSON", "replace": "real person"},
                {"find": "[PERSON]", "replace": "a person"},
                {"find": "imagined name", "replace": "[PERSON]"},
                {"find": "Alice", "replace": "[PERSON]"}]}]}, batch, 0)
        self.assertEqual(result["edits"], [{"id": "r1", "spans": [{"find": "Alice", "replace": "[PERSON]"}]}])
        self.assertEqual(len(rejected), 3)

    def test_duplicate_records_can_propose_different_spans(self):
        batch = [{"id": "r1", "text": "Alice and Bob"}]
        result, rejected = MODULE.validate_edits({"batch_id": "0", "edits": [
            {"id": "r1", "spans": [{"find": name, "replace": "[PERSON]"}]} for name in ("Alice", "Bob")]}, batch, 0)
        self.assertEqual(len(result["edits"]), 2)
        self.assertEqual(rejected, [])

    def test_rejects_wrong_batch_or_record(self):
        for parsed in ({"batch_id": "other", "edits": []},):
            with self.assertRaises(ValueError):
                MODULE.validate_edits(parsed, [{"id": "r1", "text": "coffee"}], 0)


if __name__ == "__main__":
    unittest.main()
