"""Temporal attribution-model stability diagnostics."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from itertools import combinations

import pandas as pd

from ..attribution import AttributionModel, MarkovAttribution
from ..journeys import Journey, JourneyCollection
from ..results import AuditFinding, AuditResult, AuditScore, EvidenceStatus, Severity


@dataclass(frozen=True, slots=True)
class TemporalStabilityThresholds:
    """Visible thresholds for material period-to-period drift."""

    share_volatility: float = 0.10
    high_share_volatility: float = 0.20
    rank_volatility: float = 2.0
    conversion_rate_drift: float = 0.10
    transition_stability: float = 0.80


@dataclass(frozen=True, slots=True)
class TemporalStabilityMetrics:
    """Period attribution and drift measurements."""

    period: str
    model: str
    period_summary: pd.DataFrame
    attribution: dict[str, pd.DataFrame]
    cross_period_spearman: dict[str, float]
    max_share_volatility: float
    max_rank_volatility: float
    conversion_rate_drift: float
    transition_stability: float | None


def audit_model_stability(
    journeys: JourneyCollection | tuple[Journey, ...] | list[Journey],
    *,
    model: AttributionModel,
    period: str = "W",
    thresholds: TemporalStabilityThresholds | None = None,
) -> AuditResult:
    """Rerun attribution by journey endpoint period and quantify temporal drift."""

    records = journeys.journeys if isinstance(journeys, JourneyCollection) else tuple(journeys)
    limits = thresholds or TemporalStabilityThresholds()
    normalized_period = _normalize_period(period)
    grouped: dict[str, list[Journey]] = {}
    for journey in records:
        endpoint = journey.conversion_time or journey.touchpoints[-1].timestamp
        key = str(endpoint.tz_localize(None).to_period(normalized_period))
        grouped.setdefault(key, []).append(journey)
    if len(grouped) < 2:
        explanation = (
            f"Temporal stability requires at least two {period!r} periods; "
            f"observed {len(grouped)}."
        )
        finding = AuditFinding(
            check="model_stability",
            severity=Severity.INFO,
            message=explanation,
            metric_name="period_count",
            value=len(grouped),
        )
        return AuditResult(
            "model_stability",
            (finding,),
            AuditScore(
                0.0,
                status=EvidenceStatus.INSUFFICIENT_EVIDENCE,
                explanation=explanation,
            ),
            EvidenceStatus.INSUFFICIENT_EVIDENCE,
            explanation,
        )

    attribution: dict[str, pd.DataFrame] = {}
    transitions: dict[str, pd.DataFrame] = {}
    summary_rows: list[dict[str, object]] = []
    for key, group in sorted(grouped.items()):
        fitted = deepcopy(model).fit(group)
        attribution[key] = fitted.attribute().data.copy()
        if isinstance(fitted, MarkovAttribution) and fitted.transition_probabilities is not None:
            transitions[key] = fitted.transition_probabilities.copy()
        summary_rows.append(
            {
                "period": key,
                "journeys": len(group),
                "converters": sum(journey.converted for journey in group),
                "conversion_rate": sum(journey.converted for journey in group) / len(group),
            }
        )
    summary = pd.DataFrame(
        summary_rows, columns=["period", "journeys", "converters", "conversion_rate"]
    )
    shares = _period_frame(attribution, "share")
    ranks = shares.rank(axis=0, ascending=False, method="average")
    share_volatility = _maximum_range(shares)
    rank_volatility = _maximum_range(ranks)
    correlations = _rank_correlations(ranks)
    conversion_drift = (
        float(summary["conversion_rate"].max() - summary["conversion_rate"].min())
        if not summary.empty
        else 0.0
    )
    transition_stability = _transition_stability(transitions)
    drift_flags = [
        share_volatility / limits.share_volatility,
        rank_volatility / limits.rank_volatility,
        conversion_drift / limits.conversion_rate_drift,
    ]
    if transition_stability is not None:
        drift_flags.append(
            max(0.0, limits.transition_stability - transition_stability)
            / max(limits.transition_stability, 1e-12)
        )
    maximum = max(drift_flags, default=0.0)
    if share_volatility >= limits.high_share_volatility or maximum >= 2:
        severity = Severity.HIGH
    elif maximum >= 1:
        severity = Severity.MEDIUM
    elif maximum >= 0.5:
        severity = Severity.LOW
    else:
        severity = Severity.PASS
    score = 100.0 * max(0.0, 1.0 - min(1.0, maximum / 2.0))
    metrics = TemporalStabilityMetrics(
        normalized_period,
        model.name,
        summary,
        attribution,
        correlations,
        share_volatility,
        rank_volatility,
        conversion_drift,
        transition_stability,
    )
    finding = AuditFinding(
        check="model_stability",
        severity=severity,
        score=score,
        message=(
            f"Across {len(grouped)} {period!r} periods, maximum attribution-share "
            f"volatility is {share_volatility:.1%} and conversion-rate drift is "
            f"{conversion_drift:.1%}."
        ),
        metric_name="max_share_volatility",
        value=share_volatility,
        details={"metrics": metrics, "thresholds": limits},
        recommendation=(
            "Investigate period mix, tracking changes, and model assumptions before "
            "using unstable attribution for budget decisions."
            if severity is not Severity.PASS
            else None
        ),
    )
    return AuditResult("model_stability", (finding,), AuditScore(score))


# Public descriptive alias.
audit_temporal_stability = audit_model_stability


def _normalize_period(period: str) -> str:
    aliases = {
        "week": "W",
        "weekly": "W",
        "month": "M",
        "monthly": "M",
    }
    if not isinstance(period, str) or not period.strip():
        raise ValueError("period must be a non-empty pandas period frequency.")
    return aliases.get(period.strip().lower(), period)


def _period_frame(attribution: dict[str, pd.DataFrame], value: str) -> pd.DataFrame:
    series = {
        period: frame.set_index("channel")[value].astype(float)
        for period, frame in attribution.items()
    }
    return pd.DataFrame(series).fillna(0.0).sort_index()


def _maximum_range(frame: pd.DataFrame) -> float:
    if frame.empty or frame.shape[1] < 2:
        return 0.0
    return float((frame.max(axis=1) - frame.min(axis=1)).max())


def _rank_correlations(ranks: pd.DataFrame) -> dict[str, float]:
    result: dict[str, float] = {}
    for left, right in combinations(ranks.columns, 2):
        a, b = ranks[left], ranks[right]
        if a.nunique() <= 1 or b.nunique() <= 1:
            value = 1.0 if a.equals(b) else 0.0
        else:
            value = float(a.corr(b, method="pearson"))
        result[f"{left}__{right}"] = 0.0 if pd.isna(value) else value
    return result


def _transition_stability(transitions: dict[str, pd.DataFrame]) -> float | None:
    if len(transitions) < 2:
        return None
    similarities: list[float] = []
    for left, right in combinations(transitions.values(), 2):
        states = sorted(set(left.index) | set(right.index))
        a = left.reindex(index=states, columns=states, fill_value=0.0)
        b = right.reindex(index=states, columns=states, fill_value=0.0)
        similarities.append(1.0 - float((a - b).abs().to_numpy().mean()))
    return max(0.0, min(1.0, sum(similarities) / len(similarities)))
