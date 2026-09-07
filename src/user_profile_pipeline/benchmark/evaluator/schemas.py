from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class CanonicalAnchor:
    label: str
    evidence_post_indices: list[int] = field(default_factory=list)
    source_bucket: str = "merged"  # long | short | merged


@dataclass
class CanonicalTaskView:
    task_id: str
    user_id: str
    domain: str
    domain_definition: str
    posts: list[dict[str, Any]]

    gold_status: str
    gold_interest_anchors: list[CanonicalAnchor]
    gold_representative_evidence_post_indices: list[int]
    gold_summary_natural: str
    gold_notes: str = ""


@dataclass
class CanonicalPrediction:
    status: str
    interest_anchors: list[CanonicalAnchor]
    summary_natural_pred: str
    summary_support_post_indices: list[int]


@dataclass
class PrecheckResult:
    valid: bool
    format_errors: list[str]
    cleaned_prediction: CanonicalPrediction
    hard_fail: bool


@dataclass
class TaskJudgeResult:
    task_id: str
    user_id: str
    domain: str
    gold_status: str
    pred_status: str

    precheck_valid: bool
    format_errors: list[str]

    status_correct: bool

    anchor_correctness: float
    anchor_coverage: float
    evidence_grounding: float
    summary_faithfulness: float

    judge_total_score: float
    final_task_score: float

    unsupported_anchor_labels: list[str]
    missed_gold_anchor_labels: list[str]
    unsupported_summary_claims: list[str]
    brief_rationale: str

    anchor_match_tp: int = 0
    anchor_match_fp: int = 0
    anchor_match_fn: int = 0
    anchor_match_precision: float = 0.0
    anchor_match_recall: float = 0.0
    anchor_match_f1: float = 0.0
    anchor_match_exact: bool = False
    anchor_match_pairs: list[dict[str, str]] = field(default_factory=list)
    unmatched_pred_anchor_labels: list[str] = field(default_factory=list)
    unmatched_gold_anchor_labels: list[str] = field(default_factory=list)
