from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

from .schemas import DOMAIN_NAMES, GateResult, PostObservable


_MIN_CONTENT_UNITS = 3
_URL_RE = re.compile(r"https?://\S+", re.IGNORECASE)
_WORD_RE = re.compile(r"\b[\w#@']+\b", re.UNICODE)
_MEME_HINT_RE = re.compile(
    r"\b(meme|reaction image|shitpost|template|starter pack|me irl|when you|nobody:|bro visited|caption this)\b",
    re.IGNORECASE,
)


@dataclass
class DomainConfig:
    name: str
    seed_keywords: set[str]
    seed_hashtags: set[str]


class MemeImageDetector:
    """Very conservative meme detector.

    It only returns True when metadata strongly suggests meme content.
    This avoids accidentally suppressing informative image posts.
    """

    def is_meme_image(self, post: PostObservable) -> bool:
        if not any(m.media_type == "image" for m in post.media):
            return False

        text = " ".join(filter(None, [post.text or ""] + [m.alt_text or "" for m in post.media]))
        return bool(_MEME_HINT_RE.search(text))


class NarrowGate:
    def __init__(self, domain_configs: Iterable[DomainConfig], meme_detector: MemeImageDetector | None = None, min_content_units: int = _MIN_CONTENT_UNITS) -> None:
        self.domain_configs = list(domain_configs)
        self.meme_detector = meme_detector or MemeImageDetector()
        self.min_content_units = min_content_units

    def run(self, post: PostObservable) -> GateResult:
        has_image = any(m.media_type == "image" for m in post.media)
        is_meme_image = self.meme_detector.is_meme_image(post)
        content_units = self._content_units(post)
        candidate_domains = self._candidate_domains(post)
        reasons: list[str] = []

        if has_image:
            reasons.append("has_image")
        if is_meme_image:
            reasons.append("meme_image")
        if candidate_domains:
            reasons.append("domain_cues")
        if content_units >= self.min_content_units:
            reasons.append("enough_textual_content")

        # User requirement: only judge as noise when content is too little,
        # unless there is an image, in which case it cannot be noise except meme.
        if has_image and not is_meme_image:
            return GateResult(
                post_id=post.post_id,
                is_noise=False,
                should_send_to_ai=True,
                noise_reason=None,
                has_image=has_image,
                is_meme_image=is_meme_image,
                content_units=content_units,
                gate_reason=reasons or ["has_image"],
                candidate_domains=candidate_domains,
            )

        if has_image and is_meme_image and content_units < self.min_content_units:
            return GateResult(
                post_id=post.post_id,
                is_noise=True,
                should_send_to_ai=False,
                noise_reason="meme_image_only",
                has_image=has_image,
                is_meme_image=is_meme_image,
                content_units=content_units,
                gate_reason=reasons or ["meme_image"],
                candidate_domains=candidate_domains,
            )

        if content_units < self.min_content_units:
            return GateResult(
                post_id=post.post_id,
                is_noise=True,
                should_send_to_ai=False,
                noise_reason="too_little_content",
                has_image=has_image,
                is_meme_image=is_meme_image,
                content_units=content_units,
                gate_reason=reasons or ["too_little_content"],
                candidate_domains=candidate_domains,
            )

        return GateResult(
            post_id=post.post_id,
            is_noise=False,
            should_send_to_ai=True,
            noise_reason=None,
            has_image=has_image,
            is_meme_image=is_meme_image,
            content_units=content_units,
            gate_reason=reasons or ["enough_textual_content"],
            candidate_domains=candidate_domains,
        )

    def _content_units(self, post: PostObservable) -> int:
        text = (post.text or "").strip()
        text = _URL_RE.sub(" ", text)
        word_tokens = [tok for tok in _WORD_RE.findall(text) if tok and tok not in {"#", "@"}]
        hashtag_units = len([h for h in post.hashtags if h.strip()])
        non_image_media_units = len([m for m in post.media if m.media_type in {"video", "gif"}])
        return len(word_tokens) + hashtag_units + non_image_media_units

    def _candidate_domains(self, post: PostObservable) -> list[str]:
        text = " ".join(filter(None, [post.text or "", *(m.alt_text or "" for m in post.media)]))
        lowered_text = text.lower()
        hashtags = {h.lower().lstrip("#") for h in post.hashtags}
        matched: list[str] = []
        for cfg in self.domain_configs:
            keyword_hit = any(kw in lowered_text for kw in cfg.seed_keywords)
            hashtag_hit = bool(cfg.seed_hashtags.intersection(hashtags))
            if keyword_hit or hashtag_hit:
                matched.append(cfg.name)
        return [d for d in matched if d in DOMAIN_NAMES]
