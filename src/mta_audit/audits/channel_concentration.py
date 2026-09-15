"""Channel touch, conversion, and attribution concentration diagnostics."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

import pandas as pd

from ..attribution import AttributionResult
from ..journeys import Journey, JourneyCollection
from ..results import AuditFinding, AuditResult, AuditScore, Severity


@dataclass(frozen=True, slots=True)
class ConcentrationStats:
    """Shares and standard concentration summaries for one measure."""

    shares: dict[str, float]
    hhi: float
    top1_share: float
    top3_share: float


@dataclass(frozen=True, slots=True)
class ChannelConcentrationMetrics:
    """Concentration across observed and attributed channel measures."""

    touch: ConcentrationStats
    conversion: ConcentrationStats
    attribution: dict[str, ConcentrationStats]


def calculate_channel_concentration(
    journeys: JourneyCollection | Iterable[Journey],
    attribution: Mapping[str, AttributionResult | pd.DataFrame] | None = None,
) -> ChannelConcentrationMetrics:
    """Calculate channel shares, HHI, and top-one/top-three concentration.

    Touch shares count every occurrence. Conversion shares split each converted
    journey equally across its touch occurrences. Attribution shares are read from
    each supplied model result.
    """

    records = journeys.journeys if isinstance(journeys, JourneyCollection) else tuple(journeys)
    touch_counts: Counter[str] = Counter(
        channel for journey in records for channel in journey.channels
    )
    conversion_credit: Counter[str] = Counter()
    for journey in records:
        if journey.converted:
            weight = 1.0 / len(journey.channels)
            for channel in journey.channels:
                conversion_credit[channel] += weight
    attributed: dict[str, ConcentrationStats] = {}
    for name, result in (attribution or {}).items():
        data = result.data if isinstance(result, AttributionResult) else result
        if not {"channel", "share"}.issubset(data.columns):
            raise ValueError(f"Attribution {name!r} must contain channel and share columns.")
        attributed[name] = _stats(
            data.groupby("channel", sort=False)["share"].sum().to_dict()
        )
    return ChannelConcentrationMetrics(
        touch=_stats(dict(touch_counts)),
        conversion=_stats(dict(conversion_credit)),
        attribution=attributed,
    )


def audit_channel_concentration(
    journeys: JourneyCollection | Iterable[Journey],
    attribution: Mapping[str, AttributionResult | pd.DataFrame] | None = None,
) -> AuditResult:
    """Return a structured audit based on the highest observed HHI."""

    metrics = calculate_channel_concentration(journeys, attribution)
    all_stats = [metrics.touch, metrics.conversion, *metrics.attribution.values()]
    maximum_hhi = max((stats.hhi for stats in all_stats), default=0.0)
    severity = (
        Severity.PASS
        if maximum_hhi < 0.25
        else Severity.LOW
        if maximum_hhi < 0.4
        else Severity.MEDIUM
    )
    score = 100.0 * (1.0 - maximum_hhi)
    finding = AuditFinding(
        check="channel_concentration",
        severity=severity,
        score=score,
        message=f"Highest channel HHI is {maximum_hhi:.3f}.",
        metric_name="max_hhi",
        value=maximum_hhi,
        recommendation=(
            "Review whether dominant-channel credit reflects data coverage or model bias."
            if severity is not Severity.PASS
            else None
        ),
        details={"metrics": metrics},
    )
    return AuditResult("channel_concentration", (finding,), AuditScore(score))


def _stats(values: dict[str, float]) -> ConcentrationStats:
    total = float(sum(values.values()))
    shares = {
        channel: float(value) / total if total else 0.0
        for channel, value in sorted(values.items())
    }
    descending = sorted(shares.values(), reverse=True)
    return ConcentrationStats(
        shares=shares,
        hhi=sum(share**2 for share in shares.values()),
        top1_share=sum(descending[:1]),
        top3_share=sum(descending[:3]),
    )
