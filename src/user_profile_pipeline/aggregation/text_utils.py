from __future__ import annotations

import re
import unicodedata
from collections import Counter

_WORD_RE = re.compile(r"\b\w+\b", re.UNICODE)
_PUNCT_RE = re.compile(r"[^\w\s]+", re.UNICODE)
_MULTI_SPACE_RE = re.compile(r"\s+")


def strip_accents(text: str) -> str:
    norm = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in norm if not unicodedata.combining(ch))


def normalize_free_text(text: str | None) -> str:
    raw = strip_accents((text or "").lower())
    raw = _PUNCT_RE.sub(" ", raw)
    raw = _MULTI_SPACE_RE.sub(" ", raw).strip()
    return raw


def singularize_token(token: str) -> str:
    if len(token) <= 3:
        return token
    if token.endswith("ies") and len(token) > 4:
        return token[:-3] + "y"
    if token.endswith("ses") and len(token) > 4:
        return token[:-2]
    if token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def light_lemmatize_token(token: str) -> str:
    for suffix in ("ing", "ed"):
        if token.endswith(suffix) and len(token) > len(suffix) + 2:
            base = token[: -len(suffix)]
            if len(base) >= 3:
                token = base
                break
    return singularize_token(token)


def normalize_tag(tag: str) -> str:
    text = normalize_free_text(tag).replace(" ", "_")
    parts = [light_lemmatize_token(p) for p in text.split("_") if p]
    return "_".join(parts)


def tokenize_for_dup(text: str | None) -> list[str]:
    normalized = normalize_free_text(text)
    return [light_lemmatize_token(tok) for tok in _WORD_RE.findall(normalized)]


def jaccard_similarity(a: list[str], b: list[str]) -> float:
    if not a or not b:
        return 0.0
    sa, sb = set(a), set(b)
    inter = len(sa & sb)
    union = len(sa | sb)
    return inter / union if union else 0.0


def cosine_counter(a: Counter[str], b: Counter[str]) -> float:
    if not a or not b:
        return 0.0
    keys = set(a) | set(b)
    dot = sum(a[k] * b[k] for k in keys)
    na = sum(v * v for v in a.values()) ** 0.5
    nb = sum(v * v for v in b.values()) ** 0.5
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def token_counter(text: str | None) -> Counter[str]:
    return Counter(tokenize_for_dup(text))
