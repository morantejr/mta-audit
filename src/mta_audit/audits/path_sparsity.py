"""Journey path sparsity diagnostics."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass

from ..journeys import Journey, JourneyCollection
from ..results import AuditFinding, AuditResult, AuditScore, Severity


@dataclass(frozen=True, slots=True)
class PathSparsityMetrics:
    """Summary of how thinly observations cover distinct channel paths."""

    journey_count: int
    unique_path_count: int
    unique_path_ratio: float
    singleton_path_count: int
    singleton_path_share: float
    average_journeys_per_path: float
    max_path_frequency: int


def calculate_path_sparsity(
    journeys: JourneyCollection | Iterable[Journey],
) -> PathSparsityMetrics:
    """Measure path uniqueness and frequency."""

    records = journeys.journeys if isinstance(journeys, JourneyCollection) else tuple(journeys)
    frequencies = Counter(journey.channels for journey in records)
    count = len(records)
    unique = len(frequencies)
    singletons = sum(frequency == 1 for frequency in frequencies.values())
    return PathSparsityMetrics(
        journey_count=count,
        unique_path_count=unique,
        unique_path_ratio=unique / count if count else 0.0,
        singleton_path_count=singletons,
        singleton_path_share=singletons / unique if unique else 0.0,
        average_journeys_per_path=count / unique if unique else 0.0,
        max_path_frequency=max(frequencies.values(), default=0),
    )


def audit_path_sparsity(
    journeys: JourneyCollection | Iterable[Journey],
) -> AuditResult:
    """Return a structured warning when most observed paths are singletons."""

    metrics = calculate_path_sparsity(journeys)
    ratio = metrics.singleton_path_share
    severity = (
        Severity.PASS
        if ratio <= 0.5
        else Severity.LOW
        if ratio <= 0.75
        else Severity.MEDIUM
    )
    score = 100.0 * (1.0 - ratio)
    finding = AuditFinding(
        check="path_sparsity",
        severity=severity,
        score=score,
        message=f"{ratio:.1%} of distinct paths occur only once.",
        metric_name="singleton_path_share",
        value=ratio,
        recommendation=(
            "Collect more journeys or group channels before fitting path-based models."
            if severity is not Severity.PASS
            else None
        ),
        details={"metrics": metrics},
    )
    records = journeys.journeys if isinstance(journeys, JourneyCollection) else tuple(journeys)
    unique_channels = {channel for journey in records for channel in journey.channels}
    findings = [finding]
    if len(unique_channels) > 50:
        grain_finding = AuditFinding(
            check="fine_channel_grain",
            severity=Severity.HIGH,
            score=40.0,
            message=(
                f"{len(unique_channels)} distinct channel labels look like campaign or "
                "creative grain, not media channels."
            ),
            count=len(unique_channels),
            metric_name="unique_channel_count",
            value=float(len(unique_channels)),
            recommendation=(
                "Map campaign/send names to a media-channel column before scoring mix."
            ),
        )
        findings.append(grain_finding)
        score = min(score, 40.0)
    return AuditResult("path_sparsity", tuple(findings), AuditScore(score))
