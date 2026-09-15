"""Transparent reliability scoring for attribution audits.

Only components represented by completed checks participate in the final score.
Weights are re-normalized over those available components, so a skipped check is
never interpreted as a perfect result.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, fields
from types import MappingProxyType

from .results import AuditResult, AuditScore, EvidenceStatus


@dataclass(frozen=True, slots=True)
class ScoringWeights:
    """Relative weights for the six reliability dimensions."""

    data_quality: float = 0.20
    window_stability: float = 0.15
    identity_robustness: float = 0.15
    path_quality: float = 0.15
    model_agreement: float = 0.20
    temporal_stability: float = 0.15

    def __post_init__(self) -> None:
        values = self.as_dict()
        if any(isinstance(value, bool) or value < 0 for value in values.values()):
            raise ValueError("Scoring weights must be non-negative numbers.")
        if sum(values.values()) <= 0:
            raise ValueError("At least one scoring weight must be positive.")

    def as_dict(self) -> dict[str, float]:
        """Return unnormalized configured weights."""

        return {item.name: float(getattr(self, item.name)) for item in fields(self)}

    @classmethod
    def from_value(
        cls, value: ScoringWeights | Mapping[str, float] | None
    ) -> ScoringWeights:
        """Normalize a public weight configuration."""

        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise TypeError("scoring_weights must be a ScoringWeights, mapping, or None.")
        unknown = set(value) - {item.name for item in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown scoring weights: {sorted(unknown)}")
        return cls(**value)


@dataclass(frozen=True, slots=True)
class ScoreThresholds:
    """Default interpretation bands for an overall reliability score."""

    excellent: float = 90.0
    good: float = 80.0
    caution: float = 70.0
    weak: float = 60.0

    @property
    def robust(self) -> float:
        """Qualitative-name alias for the legacy threshold field."""

        return self.excellent

    @property
    def stable(self) -> float:
        """Qualitative-name alias for the legacy threshold field."""

        return self.good

    @property
    def fragile(self) -> float:
        """Qualitative-name alias for the legacy threshold field."""

        return self.weak


# A component can be supported by multiple related checks. Their arithmetic mean
# is used before applying the component weight.
COMPONENT_CHECKS: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        "data_quality": ("data_quality", "duplicates"),
        "window_stability": ("conversion_window",),
        "identity_robustness": ("identity_loss", "touchpoint_loss", "event_loss"),
        "path_quality": ("path_sparsity", "channel_concentration"),
        "model_agreement": ("model_disagreement",),
        "temporal_stability": ("model_stability",),
    }
)


def calculate_reliability_score(
    results: Iterable[AuditResult],
    weights: ScoringWeights | Mapping[str, float] | None = None,
) -> AuditScore:
    """Return ``sum(component_score * normalized_available_weight)``.

    Each check already emits a normalized 0-100 score. Related checks are averaged
    into one component. Configured weights are then normalized over available
    components only. ``AuditScore.deductions`` records each weighted shortfall.
    """

    configured = ScoringWeights.from_value(weights).as_dict()
    by_name = {
        result.name: result.score.value
        for result in results
        if result.status is EvidenceStatus.EVALUATED
    }
    components: dict[str, float] = {}
    for component, check_names in COMPONENT_CHECKS.items():
        values = [by_name[name] for name in check_names if name in by_name]
        if values and configured[component] > 0:
            components[component] = sum(values) / len(values)
    if not components:
        # Some descriptive checks (for example exposure comparison) intentionally
        # do not claim to measure one of the six reliability dimensions.
        return AuditScore(
            0.0,
            status=EvidenceStatus.INSUFFICIENT_EVIDENCE,
            explanation="No evaluated scoring components were available.",
        )

    available_weight = sum(configured[name] for name in components)
    normalized = {name: configured[name] / available_weight for name in components}
    value = sum(components[name] * normalized[name] for name in components)
    deductions = {
        name: (100.0 - components[name]) * normalized[name] for name in components
    }
    return AuditScore(
        value=max(0.0, min(100.0, value)),
        deductions=deductions,
        components=components,
        weights=normalized,
    )


def score_label(value: float, thresholds: ScoreThresholds | None = None) -> str:
    """Return a stable human-readable interpretation for a score."""

    limits = thresholds or ScoreThresholds()
    if value >= limits.robust:
        return "Robust"
    if value >= limits.stable:
        return "Stable"
    if value >= limits.caution:
        return "Caution"
    if value >= limits.fragile:
        return "Fragile"
    return "Highly Fragile"
