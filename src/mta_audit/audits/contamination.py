"""Delayed-converter contamination diagnostics."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass

import pandas as pd

from ..results import AuditFinding, AuditResult, AuditScore, Severity
from ..schema import ColumnMapping, validate_events
from .exposure import ExposureComparisonMetrics, calculate_exposure_comparison


@dataclass(frozen=True, slots=True)
class ContaminationThresholds:
    """Visible thresholds for delayed-converter rates."""

    medium_rate: float = 0.05
    high_rate: float = 0.15
    critical_rate: float = 0.30


@dataclass(frozen=True, slots=True)
class ContaminationMetrics:
    """Counts, rates, breakdowns, and exposure comparison."""

    total_users: int
    observed_converters: int
    short_window_converters: int
    delayed_converters: int
    post_reference_converters: int
    observed_non_converters: int
    delayed_rate_all_users: float
    delayed_rate_observed_converters: float
    channel_breakdown: pd.DataFrame
    campaign_breakdown: pd.DataFrame | None
    exposure_comparison: ExposureComparisonMetrics


def audit_delayed_converter_contamination(
    events: pd.DataFrame,
    *,
    columns: ColumnMapping | dict[str, str] | None = None,
    short_window: object = 7,
    reference_window: object = 30,
    thresholds: ContaminationThresholds | None = None,
) -> AuditResult:
    """Identify users mislabeled by a short outcome window after first exposure."""

    short = _days_or_duration(short_window)
    reference = _days_or_duration(reference_window)
    if short >= reference:
        raise ValueError("short_window must be shorter than reference_window.")
    limits = thresholds or ContaminationThresholds()
    canonical, _ = validate_events(events, columns)
    counts: Counter[str] = Counter()
    channel_counts: defaultdict[str, Counter[str]] = defaultdict(Counter)
    campaign_counts: defaultdict[str, Counter[str]] = defaultdict(Counter)
    for _, user_events in canonical.groupby("user_id", sort=False):
        first_exposure = user_events["timestamp"].iloc[0]
        conversions = user_events[
            user_events["conversion"] & (user_events["timestamp"] >= first_exposure)
        ]
        if conversions.empty:
            classification = "observed_non_converter"
        else:
            outcome = conversions["timestamp"].iloc[0]
            if "conversion_timestamp" in conversions.columns:
                recorded = conversions["conversion_timestamp"].iloc[0]
                if pd.notna(recorded):
                    outcome = recorded
            delay = outcome - first_exposure
            if delay <= short:
                classification = "short_window_converter"
            elif delay <= reference:
                classification = "delayed_converter"
            else:
                classification = "post_reference_converter"
        counts[classification] += 1
        channel_counts[str(user_events["channel"].iloc[0])][classification] += 1
        if "campaign" in user_events and pd.notna(user_events["campaign"].iloc[0]):
            campaign_counts[str(user_events["campaign"].iloc[0])][classification] += 1

    total = int(sum(counts.values()))
    observed_converters = (
        counts["short_window_converter"]
        + counts["delayed_converter"]
        + counts["post_reference_converter"]
    )
    delayed = counts["delayed_converter"]
    rate_all = delayed / total if total else 0.0
    rate_converters = delayed / observed_converters if observed_converters else 0.0
    metrics = ContaminationMetrics(
        total_users=total,
        observed_converters=observed_converters,
        short_window_converters=counts["short_window_converter"],
        delayed_converters=delayed,
        post_reference_converters=counts["post_reference_converter"],
        observed_non_converters=counts["observed_non_converter"],
        delayed_rate_all_users=rate_all,
        delayed_rate_observed_converters=rate_converters,
        channel_breakdown=_breakdown(channel_counts, "channel"),
        campaign_breakdown=(
            _breakdown(campaign_counts, "campaign") if campaign_counts else None
        ),
        exposure_comparison=calculate_exposure_comparison(events, columns=columns),
    )
    if rate_all >= limits.critical_rate:
        severity = Severity.CRITICAL
    elif rate_all >= limits.high_rate:
        severity = Severity.HIGH
    elif rate_all >= limits.medium_rate:
        severity = Severity.MEDIUM
    elif delayed:
        severity = Severity.LOW
    else:
        severity = Severity.PASS
    score = 100.0 * (1.0 - min(1.0, rate_all / limits.critical_rate))
    finding = AuditFinding(
        check="converter_contamination",
        severity=severity,
        score=score,
        message=(
            f"{delayed} of {total} users ({rate_all:.1%}) convert after the short "
            f"window but within the reference window."
        ),
        count=delayed,
        metric_name="delayed_rate_all_users",
        value=rate_all,
        details={
            "metrics": metrics,
            "thresholds": limits,
            "short_window": short,
            "reference_window": reference,
        },
        recommendation=(
            "Avoid training short-window non-converters as negatives until the "
            "reference outcome window has matured."
            if delayed
            else None
        ),
    )
    return AuditResult("converter_contamination", (finding,), AuditScore(score))


def _days_or_duration(value: object) -> pd.Timedelta:
    try:
        duration = (
            pd.Timedelta(days=float(value))
            if isinstance(value, (int, float))
            else pd.Timedelta(value)
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("windows must be valid positive durations.") from exc
    if duration <= pd.Timedelta(0):
        raise ValueError("windows must be valid positive durations.")
    return duration


def _breakdown(
    values: defaultdict[str, Counter[str]], dimension: str
) -> pd.DataFrame:
    rows = []
    for key, counts in sorted(values.items()):
        total = sum(counts.values())
        rows.append(
            {
                dimension: key,
                "total_users": total,
                "short_window_converters": counts["short_window_converter"],
                "delayed_converters": counts["delayed_converter"],
                "post_reference_converters": counts["post_reference_converter"],
                "observed_non_converters": counts["observed_non_converter"],
                "delayed_rate": counts["delayed_converter"] / total if total else 0.0,
            }
        )
    return pd.DataFrame(rows)
