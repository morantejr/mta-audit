"""Absorbing Markov-chain attribution using channel removal effects."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from itertools import pairwise

import numpy as np
import pandas as pd

from ..journeys import Journey, JourneyCollection, Touchpoint
from .base import AttributionError, AttributionModel, AttributionResult, make_attribution_result

START = "START"
CONVERSION = "CONVERSION"
NULL = "NULL"
_RESERVED_STATES = {START, CONVERSION, NULL}


class MarkovAttribution(AttributionModel):
    """Attribute conversions using first-order absorbing removal effects.

    Each journey becomes ``START -> touches -> CONVERSION|NULL``. Transition
    probabilities are row-normalized observed edge counts. Removing a channel
    deletes its row and column; each predecessor reconnects across its remaining
    observed outgoing edges by row normalization. A predecessor with no remaining
    edge is sent to NULL. The removal effect is
    ``max(0, 1 - removed_probability / baseline_probability)``. Effects are
    normalized to allocate all observed conversion value and count.

    ``max_channels`` is an explicit state-space cap. When more channels are
    observed, the long tail is collapsed into ``__OTHER__``. That aggregation is
    logged on ``aggregated_channels_`` and is never applied silently.
    """

    name = "markov"

    def __init__(
        self,
        max_channels: int | None = None,
        removal_strategy: str = "reconnect",
    ) -> None:
        super().__init__()
        if max_channels is not None and max_channels <= 0:
            raise ValueError("max_channels must be a positive integer when provided.")
        if removal_strategy not in {"reconnect", "redirect_to_null"}:
            raise ValueError(
                "removal_strategy must be 'reconnect' or 'redirect_to_null'."
            )
        self.max_channels = max_channels
        self.removal_strategy = removal_strategy
        self.aggregated_channels_: tuple[str, ...] = ()
        self.transition_counts_: pd.DataFrame | None = None
        self.transition_probabilities_: pd.DataFrame | None = None
        self.baseline_conversion_probability_: float | None = None
        self.removed_conversion_probabilities_: dict[str, float] = {}
        self.removal_effects_: dict[str, float] = {}

    @property
    def transition_counts(self) -> pd.DataFrame | None:
        """Observed transition counts after fitting."""

        return self.transition_counts_

    @property
    def transition_probabilities(self) -> pd.DataFrame | None:
        """Observed row-normalized transition probabilities after fitting."""

        return self.transition_probabilities_

    @property
    def baseline_conversion_probability(self) -> float | None:
        """Probability of eventual conversion from START."""

        return self.baseline_conversion_probability_

    @property
    def removal_effects(self) -> dict[str, float]:
        """Per-channel unnormalized removal effects."""

        return dict(self.removal_effects_)

    @property
    def removed_conversion_probabilities(self) -> dict[str, float]:
        """Conversion probability after removing each channel."""

        return dict(self.removed_conversion_probabilities_)

    @property
    def transition_matrix(self) -> pd.DataFrame | None:
        """Alias for the fitted transition probability matrix."""

        return self.transition_probabilities_

    def fit(
        self, journeys: JourneyCollection | Iterable[Journey]
    ) -> MarkovAttribution:
        """Estimate transitions, baseline absorption, and channel removals."""

        super().fit(journeys)
        records = self.journeys_ or ()
        if self.max_channels is not None:
            records, self.aggregated_channels_ = _collapse_rare_channels(
                records, self.max_channels
            )
            self.journeys_ = records
        else:
            self.aggregated_channels_ = ()
        channels = sorted({channel for journey in records for channel in journey.channels})
        reserved = _RESERVED_STATES.intersection(channels)
        if reserved:
            raise AttributionError(f"Channel names conflict with Markov states: {sorted(reserved)}")
        paths = [_path(journey) for journey in records]
        self.transition_counts_, self.transition_probabilities_ = _transition_tables(paths)
        names = list(self.transition_probabilities_.index)
        matrix = self.transition_probabilities_.to_numpy(dtype=float)
        baseline = _conversion_probability_matrix(matrix, names)
        self.baseline_conversion_probability_ = baseline
        self.removed_conversion_probabilities_ = {}
        self.removal_effects_ = {}
        for channel in channels:
            reduced, reduced_names = _remove_channel_matrix(
                matrix,
                names,
                channel,
                strategy=self.removal_strategy,
            )
            probability = _conversion_probability_matrix(reduced, reduced_names)
            self.removed_conversion_probabilities_[channel] = probability
            self.removal_effects_[channel] = (
                max(0.0, 1.0 - probability / baseline) if baseline > 0 else 0.0
            )
        return self

    def _attribute_fitted(self) -> AttributionResult:
        records = self.journeys_ or ()
        channels = sorted(self.removal_effects_)
        total_value = sum(j.conversion_value for j in records if j.converted)
        total_conversions = float(sum(j.converted for j in records))
        effect_total = sum(self.removal_effects_.values())
        # Degenerate chains (for example, all paths convert regardless of channel)
        # carry no identifiable removal signal. Equal shares are explicit and stable.
        shares = {
            channel: (
                self.removal_effects_[channel] / effect_total
                if effect_total > 0
                else 1.0 / len(channels)
            )
            for channel in channels
        }
        value_credit = {channel: share * total_value for channel, share in shares.items()}
        conversion_credit = {
            channel: share * total_conversions for channel, share in shares.items()
        }
        return make_attribution_result(self.name, value_credit, conversion_credit)

    def _weights(self, journey: Journey) -> list[tuple[str, float]]:
        """Provide a valid local fallback; fitted Markov attribution uses removal effects."""

        return [(channel, 1.0) for channel in journey.channels]


# Descriptive alias commonly used by callers.
MarkovChainAttribution = MarkovAttribution


def _path(journey: Journey) -> tuple[str, ...]:
    outcome = CONVERSION if journey.converted else NULL
    return (START, *journey.channels, outcome)


def _transition_tables(paths: list[tuple[str, ...]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    counts: Counter[tuple[str, str]] = Counter()
    states = {START, CONVERSION, NULL}
    for path in paths:
        states.update(path)
        counts.update(pairwise(path))
    ordered = [START, *sorted(states - _RESERVED_STATES), CONVERSION, NULL]
    table = pd.DataFrame(0.0, index=ordered, columns=ordered)
    for (source, target), count in counts.items():
        table.loc[source, target] = float(count)
    totals = table.sum(axis=1)
    probabilities = table.div(totals.replace(0, 1), axis=0)
    # Outcome states are absorbing by model definition, rather than observations.
    probabilities.loc[CONVERSION, CONVERSION] = 1.0
    probabilities.loc[NULL, NULL] = 1.0
    return table, probabilities


def _collapse_rare_channels(
    records: tuple[Journey, ...], max_channels: int
) -> tuple[tuple[Journey, ...], tuple[str, ...]]:
    counts: Counter[str] = Counter()
    for journey in records:
        counts.update(journey.channels)
    if len(counts) <= max_channels:
        return records, ()
    keep = {channel for channel, _ in counts.most_common(max_channels - 1)}
    other = "__OTHER__"
    if other in counts and other not in keep:
        keep.add(other)
    collapsed: list[Journey] = []
    aggregated = tuple(sorted(set(counts) - keep))
    for journey in records:
        mapped = tuple(channel if channel in keep else other for channel in journey.channels)
        collapsed.append(
            Journey(
                user_id=journey.user_id,
                touchpoints=tuple(
                        Touchpoint(channel=channel, timestamp=touch.timestamp)
                    for touch, channel in zip(journey.touchpoints, mapped, strict=True)
                ),
                converted=journey.converted,
                conversion_time=journey.conversion_time,
                conversion_value=journey.conversion_value,
            )
        )
    return tuple(collapsed), aggregated


def _conversion_probability(probabilities: pd.DataFrame) -> float:
    names = list(probabilities.index)
    return _conversion_probability_matrix(probabilities.to_numpy(dtype=float), names)


def _conversion_probability_matrix(matrix: np.ndarray, names: list[str]) -> float:
    if START not in names:
        return 0.0
    absorbing = {names.index(CONVERSION), names.index(NULL)}
    transient = [index for index in range(len(names)) if index not in absorbing]
    start_pos = transient.index(names.index(START))
    conv = names.index(CONVERSION)
    q = matrix[np.ix_(transient, transient)]
    to_conversion = matrix[transient, conv]
    identity = np.eye(len(transient))
    try:
        absorption = np.linalg.solve(identity - q, to_conversion)
    except np.linalg.LinAlgError:
        absorption = np.linalg.lstsq(identity - q, to_conversion, rcond=None)[0]
    return float(np.clip(absorption[start_pos], 0.0, 1.0))


def _remove_channel_matrix(
    matrix: np.ndarray,
    names: list[str],
    channel: str,
    *,
    strategy: str = "reconnect",
) -> tuple[np.ndarray, list[str]]:
    drop = names.index(channel)
    keep = [index for index in range(len(names)) if index != drop]
    reduced = matrix[np.ix_(keep, keep)].copy()
    reduced_names = [names[index] for index in keep]
    conv = reduced_names.index(CONVERSION)
    null = reduced_names.index(NULL)
    for index, name in enumerate(reduced_names):
        if name in {CONVERSION, NULL}:
            continue
        if strategy == "redirect_to_null":
            original_index = keep[index]
            reduced[index, null] += matrix[original_index, drop]
            total = float(reduced[index].sum())
            if total > 0 and not np.isclose(total, 1.0):
                reduced[index] /= total
        else:
            total = float(reduced[index].sum())
            if total > 0:
                reduced[index] /= total
        if float(reduced[index].sum()) == 0:
            reduced[index, null] = 1.0
    reduced[conv] = 0.0
    reduced[conv, conv] = 1.0
    reduced[null] = 0.0
    reduced[null, null] = 1.0
    return reduced, reduced_names


def _remove_channel(
    probabilities: pd.DataFrame,
    channel: str,
    *,
    strategy: str = "reconnect",
) -> pd.DataFrame:
    reduced, names = _remove_channel_matrix(
        probabilities.to_numpy(dtype=float),
        list(probabilities.index),
        channel,
        strategy=strategy,
    )
    return pd.DataFrame(reduced, index=names, columns=names)
