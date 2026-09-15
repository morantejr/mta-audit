"""Observational converter versus non-converter exposure diagnostics."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import pandas as pd

from ..results import AuditFinding, AuditResult, AuditScore, Severity
from ..schema import ColumnMapping, validate_events


@dataclass(frozen=True, slots=True)
class ExposureThresholds:
    """Visible thresholds for large observational exposure differences."""

    touch_count_ratio: float = 2.0
    channel_mix_distance: float = 0.25
    high_channel_mix_distance: float = 0.50


@dataclass(frozen=True, slots=True)
class ExposureGroupStats:
    """Exposure summaries for one observed outcome group."""

    user_count: int
    touch_count_mean: float
    touch_count_median: float
    touch_count_std: float
    channel_mix: dict[str, float]
    campaign_mix: dict[str, float]
    unique_path_count: int
    path_diversity: float
    mean_recency_days: float | None
    mean_frequency: float


@dataclass(frozen=True, slots=True)
class ExposureComparisonMetrics:
    """Descriptive, non-causal differences between outcome groups."""

    converters: ExposureGroupStats
    non_converters: ExposureGroupStats
    touch_count_ratio: float | None
    channel_mix_distance: float
    campaign_mix_distance: float | None


def calculate_exposure_comparison(
    events: pd.DataFrame,
    *,
    columns: ColumnMapping | dict[str, str] | None = None,
) -> ExposureComparisonMetrics:
    """Compare pre-outcome exposure patterns without causal interpretation."""

    mapping = ColumnMapping.from_value(columns)
    canonical, _ = validate_events(events, mapping)
    groups: dict[bool, list[dict[str, object]]] = {True: [], False: []}
    for _, user_events in canonical.groupby("user_id", sort=False):
        conversions = user_events[user_events["conversion"]]
        converted = not conversions.empty
        endpoint = (
            conversions["timestamp"].iloc[0]
            if converted
            else user_events["timestamp"].iloc[-1]
        )
        exposures = user_events[user_events["timestamp"] <= endpoint]
        channels = tuple(exposures["channel"].astype(str))
        campaigns = (
            tuple(exposures["campaign"].dropna().astype(str))
            if "campaign" in exposures
            else ()
        )
        last_exposure = exposures.loc[~exposures["conversion"], "timestamp"]
        recency = (
            float((endpoint - last_exposure.iloc[-1]) / pd.Timedelta(days=1))
            if converted and not last_exposure.empty
            else None
        )
        groups[converted].append(
            {
                "channels": channels,
                "campaigns": campaigns,
                "touch_count": len(exposures),
                "recency": recency,
            }
        )
    converter_stats = _group_stats(groups[True])
    non_converter_stats = _group_stats(groups[False])
    ratio = (
        converter_stats.touch_count_mean / non_converter_stats.touch_count_mean
        if non_converter_stats.touch_count_mean > 0
        else None
    )
    channel_distance = _mix_distance(
        converter_stats.channel_mix, non_converter_stats.channel_mix
    )
    campaign_distance = (
        _mix_distance(converter_stats.campaign_mix, non_converter_stats.campaign_mix)
        if converter_stats.campaign_mix or non_converter_stats.campaign_mix
        else None
    )
    return ExposureComparisonMetrics(
        converter_stats,
        non_converter_stats,
        ratio,
        channel_distance,
        campaign_distance,
    )


def audit_exposure_comparison(
    events: pd.DataFrame,
    *,
    columns: ColumnMapping | dict[str, str] | None = None,
    thresholds: ExposureThresholds | None = None,
) -> AuditResult:
    """Report descriptive exposure imbalance with explicit non-causal language."""

    limits = thresholds or ExposureThresholds()
    metrics = calculate_exposure_comparison(events, columns=columns)
    ratio = metrics.touch_count_ratio
    ratio_size = max(ratio, 1 / ratio) if ratio and ratio > 0 else 1.0
    if metrics.channel_mix_distance >= limits.high_channel_mix_distance:
        severity = Severity.MEDIUM
    elif (
        metrics.channel_mix_distance >= limits.channel_mix_distance
        or ratio_size >= limits.touch_count_ratio
    ):
        severity = Severity.LOW
    else:
        severity = Severity.PASS
    imbalance = max(
        metrics.channel_mix_distance,
        min(1.0, abs((ratio or 1.0) - 1.0)),
    )
    score = 100.0 * (1.0 - imbalance)
    finding = AuditFinding(
        check="exposure_comparison",
        severity=severity,
        score=score,
        message=(
            "Converter versus non-converter exposure differences are observational "
            "associations only; they do not estimate causal channel effects."
        ),
        metric_name="channel_mix_distance",
        value=metrics.channel_mix_distance,
        details={"metrics": metrics, "thresholds": limits, "causal": False},
        recommendation=(
            "Treat exposure imbalance as a selection/coverage diagnostic and use a "
            "causal design before drawing incrementality conclusions."
        ),
    )
    return AuditResult("exposure_comparison", (finding,), AuditScore(score))


def _group_stats(records: list[dict[str, object]]) -> ExposureGroupStats:
    touches = pd.Series([int(record["touch_count"]) for record in records], dtype=float)
    channel_counts = Counter(
        channel for record in records for channel in record["channels"]  # type: ignore[union-attr]
    )
    campaign_counts = Counter(
        campaign for record in records for campaign in record["campaigns"]  # type: ignore[union-attr]
    )
    paths = Counter(record["channels"] for record in records)
    recencies = [float(value) for record in records if (value := record["recency"]) is not None]
    return ExposureGroupStats(
        user_count=len(records),
        touch_count_mean=float(touches.mean()) if len(touches) else 0.0,
        touch_count_median=float(touches.median()) if len(touches) else 0.0,
        touch_count_std=float(touches.std(ddof=0)) if len(touches) else 0.0,
        channel_mix=_shares(channel_counts),
        campaign_mix=_shares(campaign_counts),
        unique_path_count=len(paths),
        path_diversity=len(paths) / len(records) if records else 0.0,
        mean_recency_days=sum(recencies) / len(recencies) if recencies else None,
        mean_frequency=float(touches.mean()) if len(touches) else 0.0,
    )


def _shares(counts: Counter[str]) -> dict[str, float]:
    total = sum(counts.values())
    return {key: value / total for key, value in sorted(counts.items())} if total else {}


def _mix_distance(left: dict[str, float], right: dict[str, float]) -> float:
    keys = set(left) | set(right)
    return 0.5 * sum(abs(left.get(key, 0.0) - right.get(key, 0.0)) for key in keys)
