"""Diagnostics for disagreement among attribution models."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from math import log2
from typing import Any

import numpy as np
import pandas as pd

from ..attribution import AttributionResult
from ..results import (
    AuditFinding,
    AuditResult,
    AuditScore,
    EvidenceStatus,
    Severity,
)


@dataclass(frozen=True, slots=True)
class ModelDisagreementMetrics:
    """Channel-share and rank disagreement statistics."""

    spearman_correlations: dict[str, float]
    jensen_shannon_divergences: dict[str, float]
    max_share_difference: float
    rank_variance: float
    agreement_score: float


def calculate_model_disagreement(
    attribution: dict[str, AttributionResult | pd.DataFrame],
) -> ModelDisagreementMetrics:
    """Calculate pairwise model comparisons and a 0-100 agreement score."""

    shares = _share_frame(attribution)
    if shares.shape[1] < 2:
        return ModelDisagreementMetrics({}, {}, 0.0, 0.0, 100.0)
    spearman: dict[str, float] = {}
    js: dict[str, float] = {}
    for left, right in combinations(shares.columns, 2):
        key = f"{left}__{right}"
        left_rank = shares[left].rank(method="average")
        right_rank = shares[right].rank(method="average")
        if left_rank.nunique() == 1 or right_rank.nunique() == 1:
            correlation = 1.0 if left_rank.equals(right_rank) else 0.0
        else:
            correlation = float(left_rank.corr(right_rank, method="pearson"))
        spearman[key] = correlation
        js[key] = _jensen_shannon(shares[left].to_numpy(), shares[right].to_numpy())
    ranks = shares.rank(axis=0, ascending=False, method="average")
    rank_variance = float(ranks.var(axis=1, ddof=0).mean()) if not ranks.empty else 0.0
    max_rank_variance = max(1.0, ((len(shares) - 1) ** 2) / 4)
    max_share_difference = float((shares.max(axis=1) - shares.min(axis=1)).max())
    components = [
        (float(np.mean(list(spearman.values()))) + 1) / 2,
        1 - float(np.mean(list(js.values()))),
        1 - max_share_difference,
        1 - min(1.0, rank_variance / max_rank_variance),
    ]
    agreement = 100.0 * max(0.0, min(1.0, float(np.mean(components))))
    return ModelDisagreementMetrics(
        spearman, js, max_share_difference, rank_variance, agreement
    )


def audit_model_disagreement(
    attribution: dict[str, AttributionResult | pd.DataFrame],
) -> AuditResult:
    """Return model disagreement as a structured audit result."""

    if len(attribution) < 2:
        explanation = "Model agreement requires at least two attribution models."
        finding = AuditFinding(
            check="model_disagreement",
            severity=Severity.INFO,
            message=explanation,
            metric_name="agreement_score",
            value=None,
        )
        return AuditResult(
            "model_disagreement",
            (finding,),
            AuditScore(
                0.0,
                status=EvidenceStatus.INSUFFICIENT_EVIDENCE,
                explanation=explanation,
            ),
            EvidenceStatus.INSUFFICIENT_EVIDENCE,
            explanation,
        )
    metrics = calculate_model_disagreement(attribution)
    if metrics.agreement_score >= 80:
        severity = Severity.PASS
    elif metrics.agreement_score >= 60:
        severity = Severity.LOW
    elif metrics.agreement_score >= 40:
        severity = Severity.MEDIUM
    else:
        severity = Severity.HIGH
    finding = AuditFinding(
        check="model_disagreement",
        severity=severity,
        score=metrics.agreement_score,
        message=f"Attribution model agreement is {metrics.agreement_score:.1f}/100.",
        metric_name="agreement_score",
        value=metrics.agreement_score,
        recommendation=(
            "Investigate channels with unstable shares and validate model assumptions."
            if severity is not Severity.PASS
            else None
        ),
        details={"metrics": metrics},
    )
    return AuditResult(
        name="model_disagreement",
        findings=(finding,),
        score=AuditScore(metrics.agreement_score),
    )


def _share_frame(
    attribution: dict[str, AttributionResult | pd.DataFrame],
) -> pd.DataFrame:
    series: dict[str, pd.Series] = {}
    for name, result in attribution.items():
        data = result.data if isinstance(result, AttributionResult) else result
        if not {"channel", "share"}.issubset(data.columns):
            raise ValueError(f"Attribution {name!r} must contain channel and share columns.")
        values = data.groupby("channel", sort=False)["share"].sum().astype(float)
        total = values.sum()
        series[name] = values / total if total > 0 else values
    return pd.DataFrame(series).fillna(0.0).sort_index()


def _jensen_shannon(
    left: np.ndarray[Any, np.dtype[Any]],
    right: np.ndarray[Any, np.dtype[Any]],
) -> float:
    midpoint = (left + right) / 2

    def divergence(values: np.ndarray[Any, np.dtype[Any]]) -> float:
        return sum(
            float(value) * log2(float(value / middle))
            for value, middle in zip(values, midpoint, strict=True)
            if value > 0 and middle > 0
        )

    return float((divergence(left) + divergence(right)) / 2)
