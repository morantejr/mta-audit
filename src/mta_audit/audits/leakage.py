"""Touchpoint and event-type loss sensitivity audits."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import pandas as pd

from ..attribution import AttributionModel
from ..journeys import JourneySummary, build_journeys
from ..results import AuditFinding, AuditResult, AuditScore, Severity
from ..schema import ColumnMapping
from ..simulation import CorruptionMetadata, corrupt
from .conversion_window import resolve_attribution_model


@dataclass(frozen=True, slots=True)
class SimulationScenario:
    """One corruption level and its measured attribution response."""

    label: str
    requested_rate: float
    metadata: CorruptionMetadata
    journey_summary: JourneySummary
    attribution: Mapping[str, pd.DataFrame]
    channel_share_drift: Mapping[str, Mapping[str, float]]
    max_channel_share_drift: float
    path_length_change: float
    fragmentation_rate: float
    model_instability: float


@dataclass(frozen=True, slots=True)
class LeakageSimulationMetrics:
    """Baseline and scenario outputs for a loss sensitivity audit."""

    kind: str
    baseline_summary: JourneySummary
    baseline_attribution: Mapping[str, pd.DataFrame]
    scenarios: tuple[SimulationScenario, ...]
    max_channel_share_drift: float
    max_path_length_change: float
    max_fragmentation_rate: float
    max_model_instability: float


def audit_touchpoint_loss(
    events: pd.DataFrame,
    *,
    columns: ColumnMapping | dict[str, str] | None = None,
    rates: Iterable[float] | float = (0.05, 0.10, 0.20),
    models: Iterable[AttributionModel | str] | AttributionModel | str = ("linear",),
    lookback_window: object = "30D",
    conversion_window: object | None = None,
    include_non_converters: bool = True,
    random_state: int | None = 42,
) -> AuditResult:
    """Measure attribution drift after dropping non-conversion touchpoints."""

    normalized = _rates(rates)
    return _simulation_audit(
        "touchpoint_loss",
        events,
        columns=columns,
        scenarios=[(f"{rate:g}", {"touchpoint_loss": rate}, rate) for rate in normalized],
        models=models,
        lookback_window=lookback_window,
        conversion_window=conversion_window,
        include_non_converters=include_non_converters,
        random_state=random_state,
    )


def audit_event_loss(
    events: pd.DataFrame,
    *,
    impression_loss: float = 0,
    click_loss: float = 0,
    event_type_col: str | None = None,
    columns: ColumnMapping | dict[str, str] | None = None,
    models: Iterable[AttributionModel | str] | AttributionModel | str = ("linear",),
    lookback_window: object = "30D",
    conversion_window: object | None = None,
    include_non_converters: bool = True,
    random_state: int | None = 42,
) -> AuditResult:
    """Measure drift after independently dropping impressions and clicks."""

    scenarios = [
        (
            f"impression={impression_loss:g},click={click_loss:g}",
            {
                "impression_loss": impression_loss,
                "click_loss": click_loss,
                "event_type_col": event_type_col,
            },
            max(impression_loss, click_loss),
        )
    ]
    return _simulation_audit(
        "event_loss",
        events,
        columns=columns,
        scenarios=scenarios,
        models=models,
        lookback_window=lookback_window,
        conversion_window=conversion_window,
        include_non_converters=include_non_converters,
        random_state=random_state,
    )


def _simulation_audit(
    name: str,
    events: pd.DataFrame,
    *,
    columns: ColumnMapping | dict[str, str] | None,
    scenarios: list[tuple[str, dict[str, Any], float]],
    models: Iterable[AttributionModel | str] | AttributionModel | str,
    lookback_window: object,
    conversion_window: object | None,
    include_non_converters: bool,
    random_state: int | None,
) -> AuditResult:
    mapping = ColumnMapping.from_value(columns)
    resolved = _models(models)
    baseline = build_journeys(
        events,
        columns=mapping,
        lookback_window=lookback_window,
        conversion_window=conversion_window,
        include_non_converters=include_non_converters,
    )
    baseline_attribution = {
        model.name: deepcopy(model).attribute(baseline).data.copy() for model in resolved
    }
    original_users = set(events[mapping.user_id])
    measured: list[SimulationScenario] = []
    for index, (label, kwargs, requested_rate) in enumerate(scenarios):
        corrupted = corrupt(
            events,
            columns=mapping,
            random_state=None if random_state is None else random_state + index,
            **kwargs,
        )
        collection = build_journeys(
            corrupted.data,
            columns=mapping,
            lookback_window=lookback_window,
            conversion_window=conversion_window,
            include_non_converters=include_non_converters,
        )
        attribution = {
            model.name: deepcopy(model).attribute(collection).data.copy() for model in resolved
        }
        drift = {
            model.name: _share_drift(baseline_attribution[model.name], attribution[model.name])
            for model in resolved
        }
        maximum_drift = max(
            (abs(value) for model_drift in drift.values() for value in model_drift.values()),
            default=0.0,
        )
        synthetic_users = set(corrupted.data[mapping.user_id]) - original_users
        fragmentation = len(synthetic_users) / len(original_users) if original_users else 0.0
        path_change = (
            collection.summary.average_touches - baseline.summary.average_touches
        )
        instability = max(
            (
                sum(abs(value) for value in model_drift.values()) / 2
                for model_drift in drift.values()
            ),
            default=0.0,
        )
        measured.append(
            SimulationScenario(
                label,
                requested_rate,
                corrupted.metadata,
                collection.summary,
                attribution,
                drift,
                maximum_drift,
                path_change,
                fragmentation,
                instability,
            )
        )
    max_drift = max((item.max_channel_share_drift for item in measured), default=0.0)
    max_path = max((abs(item.path_length_change) for item in measured), default=0.0)
    max_fragmentation = max((item.fragmentation_rate for item in measured), default=0.0)
    max_instability = max((item.model_instability for item in measured), default=0.0)
    metrics = LeakageSimulationMetrics(
        name,
        baseline.summary,
        baseline_attribution,
        tuple(measured),
        max_drift,
        max_path,
        max_fragmentation,
        max_instability,
    )
    severity = (
        Severity.HIGH
        if max_drift >= 0.20
        else Severity.MEDIUM
        if max_drift >= 0.10
        else Severity.LOW
        if max_drift >= 0.05
        else Severity.PASS
    )
    score = max(0.0, 100.0 * (1.0 - min(1.0, max_drift * 2 + max_instability)))
    finding = AuditFinding(
        check=name,
        severity=severity,
        score=score,
        message=(
            f"{name.replace('_', ' ').title()} produced maximum channel-share drift "
            f"{max_drift:.1%} and path-length change {max_path:.2f}."
        ),
        metric_name="max_channel_share_drift",
        value=max_drift,
        details={"metrics": metrics},
        recommendation=(
            "Investigate tracking completeness and channel-specific event loss."
            if severity is not Severity.PASS
            else None
        ),
    )
    return AuditResult(name, (finding,), AuditScore(score))


def _rates(values: Iterable[float] | float) -> tuple[float, ...]:
    result = (values,) if isinstance(values, (int, float)) else tuple(values)
    if not result:
        raise ValueError("rates must contain at least one rate.")
    if any(isinstance(rate, bool) or not 0 <= rate <= 1 for rate in result):
        raise ValueError("rates must be between 0 and 1.")
    return tuple(float(rate) for rate in dict.fromkeys(result))


def _models(
    values: Iterable[AttributionModel | str] | AttributionModel | str,
) -> tuple[AttributionModel, ...]:
    items = (values,) if isinstance(values, (str, AttributionModel)) else tuple(values)
    if not items:
        raise ValueError("models must contain at least one model.")
    result = tuple(resolve_attribution_model(item) for item in items)
    if len({model.name for model in result}) != len(result):
        raise ValueError("models must have unique names.")
    return result


def _share_drift(baseline: pd.DataFrame, scenario: pd.DataFrame) -> dict[str, float]:
    left = baseline.set_index("channel")["share"] if not baseline.empty else pd.Series(dtype=float)
    right = scenario.set_index("channel")["share"] if not scenario.empty else pd.Series(dtype=float)
    channels = sorted(set(left.index) | set(right.index), key=str)
    return {
        str(channel): float(right.get(channel, 0.0) - left.get(channel, 0.0))
        for channel in channels
    }


