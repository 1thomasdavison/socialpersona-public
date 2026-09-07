from __future__ import annotations

import unittest
from pathlib import Path

from user_profile_pipeline.config import load_domain_configs
from user_profile_pipeline.gate import NarrowGate
from user_profile_pipeline.schemas import MediaItem, PostObservable


ROOT = Path(__file__).resolve().parents[1]


def make_post(*, text: str = "", media: list[MediaItem] | None = None) -> PostObservable:
    return PostObservable(
        user_id="test_user",
        post_id="test_post",
        platform="test",
        created_at="2026-01-01T00:00:00Z",
        post_type="post",
        text=text,
        media=media or [],
    )


class NarrowGateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.gate = NarrowGate(load_domain_configs(ROOT / "configs/domains.yaml.example"))

    def test_short_text_is_noise(self) -> None:
        result = self.gate.run(make_post(text="hello"))
        self.assertTrue(result.is_noise)
        self.assertEqual(result.noise_reason, "too_little_content")

    def test_three_content_units_are_kept_and_domain_is_detected(self) -> None:
        result = self.gate.run(make_post(text="morning trail hiking"))
        self.assertFalse(result.is_noise)
        self.assertTrue(result.should_send_to_ai)
        self.assertIn("sports_outdoor", result.candidate_domains)

    def test_non_meme_image_is_kept_even_without_text(self) -> None:
        media = [MediaItem(media_id="image_1", media_type="image", alt_text="mountain landscape")]
        result = self.gate.run(make_post(media=media))
        self.assertFalse(result.is_noise)
        self.assertTrue(result.should_send_to_ai)

    def test_meme_only_image_is_filtered(self) -> None:
        media = [MediaItem(media_id="image_1", media_type="image", alt_text="reaction image meme")]
        result = self.gate.run(make_post(media=media))
        self.assertTrue(result.is_noise)
        self.assertEqual(result.noise_reason, "meme_image_only")


if __name__ == "__main__":
    unittest.main()
