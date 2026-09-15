"""Journey-level bootstrap uncertainty for observational attribution."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .attribution import AttributionModel
from .journeys import JourneyCollection
from .results import EvidenceStatus


@dataclass(frozen=True, slots=True)
class BootstrapConfig:
    """Configuration for journey-level bootstrap resampling."""

    n_iterations: int = 200
    confidence_level: float = 0.90
    random_state: int | None = 42
    min_converted_journeys: int = 2
    stratify_conversion: bool = True

    def __post_init__(self) -> None:
        if isinstance(self.n_iterations, bool) or self.n_iterations < 10:
            raise ValueError("n_iterations must be an integer of at least 10.")
        if not 0 < self.confidence_level < 1:
            raise ValueError("confidence_level must be between 0 and 1.")
        if self.min_converted_journeys < 1:
            raise ValueError("min_converted_journeys must be at least 1.")


@dataclass(frozen=True, slots=True)
class BootstrapResult:
    """Bootstrap draws and channel-level observational stability statistics."""

    status: EvidenceStatus
    explanation: str | None
    config: BootstrapConfig
    statistics: pd.DataFrame
    draws: pd.DataFrame

    def __post_init__(self) -> None:
        object.__setattr__(self, "status", EvidenceStatus(self.status))

    @property
    def evaluated(self) -> bool:
        return self.status is EvidenceStatus.EVALUATED

    def to_dataframe(self) -> pd.DataFrame:
        """Return one row per model and channel."""

        return self.statistics.copy()


def bootstrap_attribution(
    journeys: JourneyCollection,
    models: tuple[AttributionModel, ...],
    config: BootstrapConfig | None = None,
) -> BootstrapResult:
    """Resample complete journeys and summarize attribution-share stability.

    Percentile intervals describe the attribution procedure on observed
    journeys. They are not causal confidence intervals.
    """

    settings = config or BootstrapConfig()
    records = journeys.journeys
    converted = sum(journey.converted for journey in records)
    if len(records) < 3 or converted < settings.min_converted_journeys:
        reason = (
            "Bootstrap requires at least 3 journeys and "
            f"{settings.min_converted_journeys} converted journeys; observed "
            f"{len(records)} and {converted}."
        )
        return BootstrapResult(
            EvidenceStatus.INSUFFICIENT_EVIDENCE,
            reason,
            settings,
            _empty_statistics(),
            _empty_draws(),
        )
    if not models:
        return BootstrapResult(
            EvidenceStatus.INSUFFICIENT_EVIDENCE,
            "Bootstrap requires at least one attribution model.",
            settings,
            _empty_statistics(),
            _empty_draws(),
        )

    baseline_channels = sorted(
        {
            str(channel)
            for model in models
            for channel in deepcopy(model).attribute(records).data["channel"]
        }
    )
    if not baseline_channels:
        return BootstrapResult(
            EvidenceStatus.INSUFFICIENT_EVIDENCE,
            "No attributed channels were observed in converted journeys.",
            settings,
            _empty_statistics(),
            _empty_draws(),
        )

    rng = np.random.default_rng(settings.random_state)
    rows: list[dict[str, object]] = []
    size = len(records)
    converter_records = tuple(journey for journey in records if journey.converted)
    null_records = tuple(journey for journey in records if not journey.converted)
    for iteration in range(settings.n_iterations):
        if settings.stratify_conversion and null_records:
            sample = tuple(
                converter_records[index]
                for index in rng.integers(
                    0, len(converter_records), size=len(converter_records)
                )
            ) + tuple(
                null_records[index]
                for index in rng.integers(0, len(null_records), size=len(null_records))
            )
        else:
            sample = tuple(records[index] for index in rng.integers(0, size, size=size))
        for model in models:
            frame = deepcopy(model).attribute(sample).data.set_index("channel")
            shares = pd.Series(
                {
                    channel: float(frame["share"].get(channel, 0.0))
                    for channel in baseline_channels
                }
            )
            ranks = shares.rank(ascending=False, method="min")
            for channel in baseline_channels:
                rows.append(
                    {
                        "iteration": iteration,
                        "model": model.name,
                        "channel": channel,
                        "share": float(shares[channel]),
                        "rank": float(ranks[channel]),
                    }
                )
    draws = pd.DataFrame(rows, columns=_empty_draws().columns)
    alpha = (1.0 - settings.confidence_level) / 2.0
    grouped = draws.groupby(["model", "channel"], sort=True)
    statistics = grouped.agg(
        mean_share=("share", "mean"),
        median_share=("share", "median"),
        share_std=("share", "std"),
        interval_low=("share", lambda values: values.quantile(alpha)),
        interval_high=("share", lambda values: values.quantile(1 - alpha)),
        median_rank=("rank", "median"),
        rank_low=("rank", lambda values: values.quantile(alpha)),
        rank_high=("rank", lambda values: values.quantile(1 - alpha)),
        probability_top_1=("rank", lambda values: float((values <= 1).mean())),
        probability_top_2=("rank", lambda values: float((values <= 2).mean())),
        probability_top_3=("rank", lambda values: float((values <= 3).mean())),
    ).reset_index()
    statistics.insert(2, "confidence_level", settings.confidence_level)
    return BootstrapResult(
        EvidenceStatus.EVALUATED,
        None,
        settings,
        statistics,
        draws,
    )


def _empty_draws() -> pd.DataFrame:
    return pd.DataFrame(columns=["iteration", "model", "channel", "share", "rank"])


def _empty_statistics() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "model",
            "channel",
            "confidence_level",
            "mean_share",
            "median_share",
            "share_std",
            "interval_low",
            "interval_high",
            "median_rank",
            "rank_low",
            "rank_high",
            "probability_top_1",
            "probability_top_2",
            "probability_top_3",
        ]
    )
