"""Assumption sensitivity and decision-robustness summaries."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import pandas as pd

from .attribution import AttributionResult
from .bootstrap import BootstrapResult
from .results import AuditResult, EvidenceStatus


@dataclass(frozen=True, slots=True)
class SensitivityResult:
    """Attribution outputs under named observational assumptions."""

    status: EvidenceStatus
    explanation: str | None
    baseline: str | None
    scenarios: dict[str, pd.DataFrame]

    def __post_init__(self) -> None:
        object.__setattr__(self, "status", EvidenceStatus(self.status))
        object.__setattr__(self, "scenarios", MappingProxyType(dict(self.scenarios)))

    def matrix(self, metric: str = "rank") -> pd.DataFrame:
        """Return channels by scenario for attribution rank or share."""

        if metric not in {"rank", "share"}:
            raise ValueError("metric must be 'rank' or 'share'.")
        if not self.scenarios:
            return pd.DataFrame(columns=["channel"])
        valid = {
            label: frame
            for label, frame in self.scenarios.items()
            if _valid_attribution(frame)
        }
        channels = sorted(
            {
                str(channel)
                for frame in valid.values()
                for channel in frame.get("channel", pd.Series(dtype=str))
            }
        )
        result = pd.DataFrame({"channel": channels})
        for label, frame in valid.items():
            indexed = frame.set_index("channel")
            if metric == "share":
                values = indexed["share"].astype(float)
                result[label] = result["channel"].map(values).fillna(0.0)
            else:
                shares = indexed["share"].astype(float).reindex(channels, fill_value=0.0)
                ranks = shares.rank(ascending=False, method="min")
                result[label] = result["channel"].map(ranks).astype(float)
        return result


@dataclass(frozen=True, slots=True)
class DecisionRobustnessResult:
    """Channel rankings across tested assumptions and bootstrap samples."""

    status: EvidenceStatus
    explanation: str | None
    statistics: pd.DataFrame
    interpretation: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "status", EvidenceStatus(self.status))

    def to_dataframe(self) -> pd.DataFrame:
        return self.statistics.copy()

    def __str__(self) -> str:
        return self.interpretation


def collect_sensitivity(
    attribution: dict[str, AttributionResult],
    results: tuple[AuditResult, ...],
    additional: dict[str, pd.DataFrame] | None = None,
) -> SensitivityResult:
    """Collect model, window, temporal, and loss scenarios already computed."""

    preferred = "linear" if "linear" in attribution else next(iter(attribution), None)
    scenarios: dict[str, pd.DataFrame] = {}
    if preferred is not None:
        scenarios["baseline"] = attribution[preferred].data.copy()
    for name, output in attribution.items():
        if name != preferred:
            scenarios[f"model:{name}"] = output.data.copy()
    scenarios.update({label: frame.copy() for label, frame in (additional or {}).items()})

    for result in results:
        for finding in result.findings:
            metrics = finding.details.get("metrics")
            if metrics is None:
                continue
            nested = getattr(metrics, "attribution", None)
            if isinstance(nested, Mapping):
                for label, frame in nested.items():
                    if isinstance(frame, pd.DataFrame):
                        scenarios[f"{result.name}:{label}"] = frame.copy()
            for scenario in getattr(metrics, "scenarios", ()):
                outputs = getattr(scenario, "attribution", {})
                if preferred in outputs:
                    scenarios[f"{result.name}:{scenario.label}"] = outputs[preferred].copy()

    scenarios = {
        label: frame for label, frame in scenarios.items() if _valid_attribution(frame)
    }
    distinct = _distinct_scenarios(scenarios)
    if len(distinct) < 2:
        return SensitivityResult(
            EvidenceStatus.INSUFFICIENT_EVIDENCE,
            "At least two distinct attribution scenarios are required.",
            "baseline" if "baseline" in scenarios else next(iter(scenarios), None),
            scenarios,
        )
    return SensitivityResult(
        EvidenceStatus.EVALUATED,
        None,
        "baseline" if "baseline" in scenarios else next(iter(scenarios), None),
        scenarios,
    )


def evaluate_decision_robustness(
    sensitivity: SensitivityResult,
    bootstrap: BootstrapResult | None = None,
) -> DecisionRobustnessResult:
    """Summarize whether channel ordering survives tested assumptions."""

    rank_matrix = sensitivity.matrix("rank")
    share_matrix = sensitivity.matrix("share")
    if sensitivity.status is not EvidenceStatus.EVALUATED or len(rank_matrix) < 2:
        reason = sensitivity.explanation or "At least two channels are required."
        return DecisionRobustnessResult(
            EvidenceStatus.INSUFFICIENT_EVIDENCE,
            reason,
            _empty_decision_frame(),
            f"Decision robustness was not evaluated: {reason}",
        )

    scenario_rank = rank_matrix.set_index("channel")
    scenario_share = share_matrix.set_index("channel")
    channel_count = len(scenario_rank)
    preferred_model = None
    if bootstrap is not None and bootstrap.evaluated and not bootstrap.draws.empty:
        preferred_model = (
            "linear"
            if "linear" in set(bootstrap.draws["model"])
            else str(bootstrap.draws["model"].iloc[0])
        )
    rows: list[dict[str, Any]] = []
    for channel in scenario_rank.index:
        ranks = scenario_rank.loc[channel].astype(float)
        shares = scenario_share.loc[channel].astype(float)
        bootstrap_ranks = pd.Series(dtype=float)
        if preferred_model is not None:
            bootstrap_ranks = bootstrap.draws.loc[
                (bootstrap.draws["model"] == preferred_model)
                & (bootstrap.draws["channel"] == channel),
                "rank",
            ].astype(float)
        top_1 = float((ranks <= 1).mean())
        top_2 = float((ranks <= 2).mean())
        top_3 = float((ranks <= 3).mean())
        low = float(ranks.quantile(0.05))
        high = float(ranks.quantile(0.95))
        bootstrap_top_1 = (
            float((bootstrap_ranks <= 1).mean()) if len(bootstrap_ranks) else None
        )
        bootstrap_top_2 = (
            float((bootstrap_ranks <= 2).mean()) if len(bootstrap_ranks) else None
        )
        bootstrap_top_3 = (
            float((bootstrap_ranks <= 3).mean()) if len(bootstrap_ranks) else None
        )
        classification = _classification(
            top_1,
            top_2,
            high - low,
            channel_count,
        )
        rows.append(
            {
                "channel": channel,
                "baseline_rank": float(ranks.get("baseline", ranks.iloc[0])),
                "median_rank": float(ranks.median()),
                "rank_low": low,
                "rank_high": high,
                "probability_top_1": top_1,
                "probability_top_2": top_2,
                "probability_top_3": top_3,
                "bootstrap_median_rank": (
                    float(bootstrap_ranks.median()) if len(bootstrap_ranks) else None
                ),
                "bootstrap_rank_low": (
                    float(bootstrap_ranks.quantile(0.05))
                    if len(bootstrap_ranks)
                    else None
                ),
                "bootstrap_rank_high": (
                    float(bootstrap_ranks.quantile(0.95))
                    if len(bootstrap_ranks)
                    else None
                ),
                "bootstrap_probability_top_1": bootstrap_top_1,
                "bootstrap_probability_top_2": bootstrap_top_2,
                "bootstrap_probability_top_3": bootstrap_top_3,
                "share_min": float(shares.min()),
                "share_max": float(shares.max()),
                "ranking_volatility": float(ranks.std(ddof=0)),
                "classification": classification,
            }
        )
    statistics = pd.DataFrame(rows).sort_values(
        ["baseline_rank", "channel"], ignore_index=True
    )
    interpretation = _interpret(statistics)
    return DecisionRobustnessResult(
        EvidenceStatus.EVALUATED,
        None,
        statistics,
        interpretation,
    )


def _distinct_scenarios(scenarios: dict[str, pd.DataFrame]) -> set[tuple[tuple[str, float], ...]]:
    return {
        tuple(
            sorted(
                (str(row.channel), round(float(row.share), 12))
                for row in frame[["channel", "share"]].itertuples(index=False)
            )
        )
        for frame in scenarios.values()
        if _valid_attribution(frame)
    }


def _classification(
    probability_top_1: float,
    probability_top_2: float,
    rank_span: float,
    channel_count: int,
) -> str:
    if channel_count < 3:
        if probability_top_1 >= 0.90 and rank_span == 0:
            return "Robust"
        if probability_top_1 >= 0.75 and rank_span <= 1:
            return "Stable"
        if probability_top_1 >= 0.50:
            return "Uncertain"
        return "Fragile"
    if (
        probability_top_1 >= 0.75
        and probability_top_2 >= 0.90
        and rank_span <= 1
    ):
        return "Robust"
    if (
        probability_top_1 >= 0.50
        and probability_top_2 >= 0.75
        and rank_span <= 2
    ):
        return "Stable"
    if probability_top_2 >= 0.50:
        return "Uncertain"
    return "Fragile"


def _interpret(statistics: pd.DataFrame) -> str:
    leader = statistics.iloc[0]
    lines = [
        (
            f"{leader['channel']} remains the top-ranked channel in "
            f"{float(leader['probability_top_1']):.0%} and top-two in "
            f"{float(leader['probability_top_2']):.0%} of tested assumption scenarios."
        )
    ]
    if pd.notna(leader.get("bootstrap_probability_top_2")):
        lines.append(
            f"In journey bootstrap samples, {leader['channel']} is top-two in "
            f"{float(leader['bootstrap_probability_top_2']):.0%} of draws."
        )
    unstable = statistics.loc[
        statistics["classification"].isin(["Uncertain", "Fragile"]), "channel"
    ].astype(str)
    if len(unstable):
        lines.append(
            f"{', '.join(unstable[:3])} frequently change position; budget "
            "reallocation among them is not strongly supported by this "
            "observational attribution analysis alone."
        )
    lines.append(
        "Frequencies describe stability across tested assumptions, not causal truth."
    )
    return "\n".join(lines)


def _empty_decision_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "channel",
            "baseline_rank",
            "median_rank",
            "rank_low",
            "rank_high",
            "probability_top_1",
            "probability_top_2",
            "probability_top_3",
            "bootstrap_median_rank",
            "bootstrap_rank_low",
            "bootstrap_rank_high",
            "bootstrap_probability_top_1",
            "bootstrap_probability_top_2",
            "bootstrap_probability_top_3",
            "share_min",
            "share_max",
            "ranking_volatility",
            "classification",
        ]
    )


def empty_decision_frame() -> pd.DataFrame:
    """Return the stable schema for an unevaluated decision result."""

    return _empty_decision_frame()


def _valid_attribution(frame: pd.DataFrame) -> bool:
    if frame.empty or not {"channel", "share"}.issubset(frame.columns):
        return False
    shares = pd.to_numeric(frame["share"], errors="coerce").fillna(0.0)
    return bool((shares > 0).any() and shares.sum() > 0)
