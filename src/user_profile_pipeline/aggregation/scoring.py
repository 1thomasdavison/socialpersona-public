from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from statistics import mean


@dataclass
class DomainSupportStats:
    domain: str
    support_posts: list[str]
    weak_posts: list[str]
    effective_support_posts: float
    mean_confidence: float
    distinct_time_bins: int
    recent_90_support: float
    recent_30_support: float
    recent_90_mean_confidence: float
    stable_score: float
    recent_score: float
    status: str


@dataclass
class InterestSupportStats:
    interest: str
    support_posts: list[str]
    effective_support_posts: float
    mean_confidence: float
    distinct_time_bins: int
    recent_90_support: float
    recent_score: float
    stable_score: float
    status: str


class TemporalScorer:
    def __init__(self, support_threshold: float = 0.60, weak_threshold: float = 0.45) -> None:
        self.support_threshold = support_threshold
        self.weak_threshold = weak_threshold

    def compute_domain_stats(self, domain: str, records: list[dict], window_days: int, observed_end: datetime) -> DomainSupportStats:
        support = [r for r in records if r["confidence"] >= self.support_threshold]
        weak = [r for r in records if self.weak_threshold <= r["confidence"] < self.support_threshold]
        effective_support = sum(r["dup_weight"] for r in support)
        mean_conf = mean([r["confidence"] for r in support]) if support else 0.0
        bins = self._distinct_bins([r["created_at"] for r in support], window_days)
        recent_90 = [r for r in support if r["created_at"] >= observed_end - timedelta(days=90)]
        recent_30 = [r for r in support if r["created_at"] >= observed_end - timedelta(days=30)]
        recent_90_support = sum(r["dup_weight"] for r in recent_90)
        recent_30_support = sum(r["dup_weight"] for r in recent_30)
        recent_90_mean_conf = mean([r["confidence"] for r in recent_90]) if recent_90 else 0.0
        stable_score = round(
            0.45 * min(1.0, effective_support / 8.0)
            + 0.35 * min(1.0, bins / self.required_bins(window_days))
            + 0.20 * mean_conf,
            6,
        )
        recent_score = round(
            0.50 * min(1.0, recent_90_support / 5.0)
            + 0.20 * min(1.0, recent_30_support / 3.0)
            + 0.30 * recent_90_mean_conf,
            6,
        )
        return DomainSupportStats(
            domain=domain,
            support_posts=sorted({r["post_id"] for r in support}),
            weak_posts=sorted({r["post_id"] for r in weak}),
            effective_support_posts=round(effective_support, 6),
            mean_confidence=round(mean_conf, 6),
            distinct_time_bins=bins,
            recent_90_support=round(recent_90_support, 6),
            recent_30_support=round(recent_30_support, 6),
            recent_90_mean_confidence=round(recent_90_mean_conf, 6),
            stable_score=stable_score,
            recent_score=recent_score,
            status=self.classify_domain_status(stable_score, recent_score, bool(support or weak)),
        )

    def compute_interest_stats(self, interest: str, records: list[dict], window_days: int, observed_end: datetime) -> InterestSupportStats:
        support = [r for r in records if r["confidence"] >= self.support_threshold]
        effective_support = sum(r["dup_weight"] for r in support)
        mean_conf = mean([r["confidence"] for r in support]) if support else 0.0
        bins = self._distinct_bins([r["created_at"] for r in support], window_days)
        recent_90 = [r for r in support if r["created_at"] >= observed_end - timedelta(days=90)]
        recent_90_support = sum(r["dup_weight"] for r in recent_90)
        recent_score = round(
            0.65 * min(1.0, recent_90_support / 2.0)
            + 0.35 * (mean([r["confidence"] for r in recent_90]) if recent_90 else 0.0),
            6,
        )
        stable_score = round(
            0.60 * min(1.0, effective_support / 3.0)
            + 0.20 * min(1.0, bins / max(2, self.required_bins(window_days) - 1))
            + 0.20 * mean_conf,
            6,
        )
        status = "weak_signal"
        if stable_score >= 0.62 and len({r["post_id"] for r in support}) >= 3 and bins >= 2:
            status = "stable"
        elif recent_score >= 0.60 and len({r["post_id"] for r in recent_90}) >= 2:
            status = "recent"
        return InterestSupportStats(
            interest=interest,
            support_posts=sorted({r["post_id"] for r in support}),
            effective_support_posts=round(effective_support, 6),
            mean_confidence=round(mean_conf, 6),
            distinct_time_bins=bins,
            recent_90_support=round(recent_90_support, 6),
            recent_score=recent_score,
            stable_score=stable_score,
            status=status,
        )

    @staticmethod
    def required_bins(window_days: int) -> int:
        if window_days < 90:
            return 2
        if window_days < 365:
            return 3
        return 4

    @staticmethod
    def classify_domain_status(stable_score: float, recent_score: float, has_any_evidence: bool) -> str:
        if stable_score >= 0.65 and recent_score >= 0.65:
            return "stable_with_recent_surge"
        if stable_score >= 0.65 and recent_score < 0.65:
            return "stable"
        if recent_score >= 0.65 and stable_score < 0.65:
            return "recent"
        if has_any_evidence:
            return "weak_signal"
        return "absent"

    @staticmethod
    def _distinct_bins(dates: list[datetime], window_days: int) -> int:
        if not dates:
            return 0
        if window_days >= 180:
            bins = {f"{d.year:04d}-{d.month:02d}" for d in dates}
        else:
            bins = {f"{d.isocalendar().year:04d}-W{d.isocalendar().week:02d}" for d in dates}
        return len(bins)
