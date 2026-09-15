"""Sensitivity analysis for attribution conversion-window choices."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from itertools import combinations

import pandas as pd

from ..attribution import (
    AttributionModel,
    FirstTouchAttribution,
    LastTouchAttribution,
    LinearAttribution,
    MarkovAttribution,
)
from ..journeys import build_journeys
from ..results import AuditFinding, AuditResult, AuditScore, EvidenceStatus, Severity
from ..schema import ColumnMapping


@dataclass(frozen=True, slots=True)
class ConversionWindowThresholds:
    """Visible thresholds for conversion-window sensitivity."""

    low_spearman: float = 0.90
    high_spearman: float = 0.70
    share_volatility: float = 0.10
    high_share_volatility: float = 0.20
    high_risk_share_change: float = 0.10
    high_risk_rank_change: float = 2.0


@dataclass(frozen=True, slots=True)
class ConversionWindowMetrics:
    """Attribution outputs and cross-window sensitivity measures."""

    windows: tuple[str, ...]
    model: str
    attribution: dict[str, pd.DataFrame]
    channel_comparison: pd.DataFrame
    cross_window_spearman: dict[str, float]
    max_share_volatility: float
    high_risk_channels: tuple[str, ...]


def audit_conversion_window(
    events: pd.DataFrame,
    *,
    columns: ColumnMapping | dict[str, str] | None = None,
    windows: tuple[object, ...] | list[object] = (7, 14, 30),
    model: AttributionModel | str = "linear",
    lookback_window: object = "30D",
    thresholds: ConversionWindowThresholds | None = None,
) -> AuditResult:
    """Rebuild journeys and rerun one model for every candidate window."""

    limits = thresholds or ConversionWindowThresholds()
    normalized_windows = _normalize_windows(windows)
    if len(normalized_windows) < 2:
        explanation = (
            "Conversion-window stability requires at least two distinct windows."
        )
        finding = AuditFinding(
            check="conversion_window",
            severity=Severity.INFO,
            message=explanation,
            metric_name="window_count",
            value=len(normalized_windows),
        )
        return AuditResult(
            "conversion_window",
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
    for label, duration in normalized_windows:
        collection = build_journeys(
            events,
            columns=columns,
            lookback_window=lookback_window,
            conversion_window=duration,
            include_non_converters=True,
        )
        result = deepcopy(resolve_attribution_model(model)).attribute(collection)
        attribution[label] = result.data.copy()

    comparison = _comparison_frame(attribution)
    correlations = _cross_window_spearman(comparison, tuple(attribution))
    volatility = (
        float(comparison.filter(like="share__").max(axis=1).sub(
            comparison.filter(like="share__").min(axis=1)
        ).max())
        if not comparison.empty
        else 0.0
    )
    risk = tuple(
        comparison.loc[
            (comparison.filter(like="share__").max(axis=1) -
             comparison.filter(like="share__").min(axis=1)
             >= limits.high_risk_share_change)
            | (
                comparison.filter(like="rank__").max(axis=1) -
                comparison.filter(like="rank__").min(axis=1)
                >= limits.high_risk_rank_change
            ),
            "channel",
        ].astype(str)
    )
    minimum_correlation = min(correlations.values(), default=1.0)
    if (
        minimum_correlation < limits.high_spearman
        or volatility >= limits.high_share_volatility
    ):
        severity = Severity.HIGH
    elif minimum_correlation < limits.low_spearman or volatility >= limits.share_volatility:
        severity = Severity.MEDIUM
    elif risk:
        severity = Severity.LOW
    else:
        severity = Severity.PASS
    correlation_component = (minimum_correlation + 1.0) / 2.0
    score = 100.0 * max(0.0, min(1.0, 0.5 * correlation_component + 0.5 * (1 - volatility)))
    metrics = ConversionWindowMetrics(
        tuple(attribution), resolve_attribution_model(model).name, attribution, comparison,
        correlations, volatility, risk
    )
    finding = AuditFinding(
        check="conversion_window",
        severity=severity,
        score=score,
        message=(
            f"Attribution across {len(attribution)} windows has minimum rank "
            f"correlation {minimum_correlation:.3f} and maximum share volatility "
            f"{volatility:.1%}."
        ),
        metric_name="max_share_volatility",
        value=volatility,
        count=len(risk),
        details={"metrics": metrics, "thresholds": limits},
        recommendation=(
            "Review high-risk channels and justify the production conversion window."
            if severity is not Severity.PASS
            else None
        ),
    )
    return AuditResult("conversion_window", (finding,), AuditScore(score))


def _normalize_windows(
    windows: tuple[object, ...] | list[object],
) -> list[tuple[str, pd.Timedelta]]:
    if not windows:
        raise ValueError("windows must contain at least one positive duration.")
    result: list[tuple[str, pd.Timedelta]] = []
    seen: set[pd.Timedelta] = set()
    for value in windows:
        duration = (
            pd.Timedelta(days=float(value))
            if isinstance(value, (int, float))
            else pd.Timedelta(value)
        )
        if duration <= pd.Timedelta(0):
            raise ValueError("windows must contain only positive durations.")
        if duration in seen:
            continue
        seen.add(duration)
        result.append((_duration_label(duration), duration))
    return result


def _duration_label(value: pd.Timedelta) -> str:
    days = value / pd.Timedelta(days=1)
    return f"{int(days) if float(days).is_integer() else days:g}D"


def resolve_attribution_model(model: AttributionModel | str) -> AttributionModel:
    """Resolve a built-in model name or validate a model instance."""
    if isinstance(model, AttributionModel):
        return model
    models: dict[str, type[AttributionModel]] = {
        "first_touch": FirstTouchAttribution,
        "last_touch": LastTouchAttribution,
        "linear": LinearAttribution,
        "markov": MarkovAttribution,
        "markov_chain": MarkovAttribution,
    }
    try:
        return models[model.strip().lower()]()
    except (AttributeError, KeyError) as exc:
        raise ValueError(f"Unknown attribution model: {model!r}") from exc


def _comparison_frame(attribution: dict[str, pd.DataFrame]) -> pd.DataFrame:
    channels = sorted(
        {
            str(channel)
            for frame in attribution.values()
            for channel in frame["channel"]
        }
    )
    result = pd.DataFrame({"channel": channels})
    for label, frame in attribution.items():
        indexed = frame.set_index("channel")
        result[f"credit__{label}"] = (
            result["channel"].map(indexed["attribution_credit"]).fillna(0.0)
        )
        result[f"share__{label}"] = result["channel"].map(indexed["share"]).fillna(0.0)
        ranks = indexed["attribution_credit"].rank(ascending=False, method="min")
        result[f"rank__{label}"] = result["channel"].map(ranks).fillna(len(channels) + 1)
    if attribution:
        first, last = next(iter(attribution)), next(reversed(attribution))
        result["absolute_credit_change"] = result[f"credit__{last}"] - result[f"credit__{first}"]
        denominator = result[f"credit__{first}"].abs()
        result["percentage_credit_change"] = (
            result["absolute_credit_change"].div(denominator.where(denominator > 0))
        )
        result["rank_change"] = result[f"rank__{last}"] - result[f"rank__{first}"]
        result["share_volatility"] = (
            result.filter(like="share__").max(axis=1) - result.filter(like="share__").min(axis=1)
        )
    return result


def _cross_window_spearman(
    comparison: pd.DataFrame, labels: tuple[str, ...]
) -> dict[str, float]:
    values: dict[str, float] = {}
    for left, right in combinations(labels, 2):
        a, b = comparison[f"rank__{left}"], comparison[f"rank__{right}"]
        if a.nunique() <= 1 or b.nunique() <= 1:
            correlation = 1.0 if a.equals(b) else 0.0
        else:
            correlation = float(a.corr(b, method="pearson"))
        values[f"{left}__{right}"] = 0.0 if pd.isna(correlation) else correlation
    return values
