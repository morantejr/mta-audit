"""Customer journey records and construction."""

from __future__ import annotations

from collections import Counter
from collections.abc import Hashable, Iterable, Mapping
from dataclasses import dataclass, field
from datetime import timedelta
from types import MappingProxyType

import numpy as np
import pandas as pd

from .schema import ColumnMapping, ValidationSummary, validate_events


class JourneyError(ValueError):
    """Raised for invalid journey-builder configuration."""


@dataclass(frozen=True, slots=True)
class Touchpoint:
    """One ordered channel interaction."""

    channel: str
    timestamp: pd.Timestamp


@dataclass(frozen=True, slots=True)
class Journey:
    """An ordered sequence ending in a conversion or observation boundary."""

    user_id: Hashable
    touchpoints: tuple[Touchpoint, ...]
    converted: bool
    conversion_time: pd.Timestamp | None = None
    conversion_value: float = 1.0

    def __post_init__(self) -> None:
        if not self.touchpoints:
            raise JourneyError("A journey must contain at least one touchpoint.")
        timestamps = [touch.timestamp for touch in self.touchpoints]
        if timestamps != sorted(timestamps):
            raise JourneyError("Touchpoints must be in chronological order.")
        if self.converted and self.conversion_time is None:
            raise JourneyError("Converted journeys require conversion_time.")
        if not self.converted and self.conversion_time is not None:
            raise JourneyError("Non-converting journeys cannot have conversion_time.")

    @property
    def channels(self) -> tuple[str, ...]:
        """Return channels in order, preserving repeats and direct traffic."""

        return tuple(touch.channel for touch in self.touchpoints)

    @property
    def first_channel(self) -> str:
        """Return the first channel."""

        return self.touchpoints[0].channel

    @property
    def last_channel(self) -> str:
        """Return the last channel."""

        return self.touchpoints[-1].channel

    @property
    def duration(self) -> pd.Timedelta:
        """Return elapsed time between first and last touch."""

        return self.touchpoints[-1].timestamp - self.touchpoints[0].timestamp


@dataclass(frozen=True, slots=True)
class JourneySummary:
    """Aggregate journey-building statistics."""

    journey_count: int
    converter_count: int
    non_converter_count: int
    touchpoint_count: int
    unique_channel_count: int
    average_touches: float
    median_touches: float
    minimum_touches: int
    maximum_touches: int
    conversion_rate: float
    average_duration: pd.Timedelta
    unique_path_count: int
    converter_average_path_length: float = 0.0
    non_converter_average_path_length: float = 0.0
    converter_median_path_length: float = 0.0
    non_converter_median_path_length: float = 0.0
    touchpoint_frequency: Mapping[str, int] = field(default_factory=dict)
    channel_frequency: Mapping[str, int] = field(default_factory=dict)
    repeated_touch_rate: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "touchpoint_frequency", MappingProxyType(dict(self.touchpoint_frequency))
        )
        object.__setattr__(
            self, "channel_frequency", MappingProxyType(dict(self.channel_frequency))
        )

    @property
    def total_journeys(self) -> int:
        """Alias for journey_count."""

        return self.journey_count

    @property
    def path_count(self) -> int:
        """Alias for journey_count."""

        return self.journey_count

    @property
    def total_touchpoints(self) -> int:
        """Alias for touchpoint_count."""

        return self.touchpoint_count

    @property
    def average_path_length(self) -> float:
        """Alias for average_touches."""

        return self.average_touches

    @property
    def median_path_length(self) -> float:
        """Alias for median_touches."""

        return self.median_touches


@dataclass(frozen=True, slots=True)
class JourneyCollection:
    """Built journeys plus source validation and journey summaries."""

    journeys: tuple[Journey, ...]
    summary: JourneySummary
    validation: ValidationSummary

    def __iter__(self) -> Iterable[Journey]:
        return iter(self.journeys)

    def __len__(self) -> int:
        return len(self.journeys)


def build_journeys(
    events: pd.DataFrame,
    *,
    columns: ColumnMapping | dict[str, str] | None = None,
    lookback_window: str | timedelta | pd.Timedelta = "30D",
    conversion_window: str | timedelta | pd.Timedelta | None = None,
    include_non_converters: bool = True,
) -> JourneyCollection:
    """Build journeys from event-level data.

    A converted user yields one journey per conversion. Touches already assigned to
    an earlier conversion are not reused. ``lookback_window`` limits history before
    each endpoint. ``conversion_window``, when set, additionally limits a converting
    path's age. Non-converters produce one journey ending at their final observation.
    """

    lookback = _positive_timedelta(lookback_window, "lookback_window")
    conversion_limit = (
        _positive_timedelta(conversion_window, "conversion_window")
        if conversion_window is not None
        else None
    )
    canonical, validation = validate_events(events, columns)
    journeys = _build_from_canonical(
        canonical,
        lookback=lookback,
        conversion_limit=conversion_limit,
        include_non_converters=include_non_converters,
    )
    summary = summarize_journeys(journeys)
    return JourneyCollection(tuple(journeys), summary, validation)


class JourneyBuilder:
    """Configurable builder for the public journey-construction API."""

    def __init__(
        self,
        data: pd.DataFrame,
        *,
        columns: ColumnMapping | dict[str, str] | None = None,
        user_col: str | None = None,
        timestamp_col: str | None = None,
        channel_col: str | None = None,
        conversion_col: str | None = None,
        lookback_window: str | timedelta | pd.Timedelta = "30D",
        conversion_window: str | timedelta | pd.Timedelta | None = None,
        include_non_converters: bool = True,
    ) -> None:
        if columns is not None and any(
            value is not None
            for value in (user_col, timestamp_col, channel_col, conversion_col)
        ):
            raise ValueError("Use either columns or individual *_col arguments, not both.")
        self.data = data
        self.columns = ColumnMapping.from_value(
            columns
            or {
                key: value
                for key, value in {
                    "user_id": user_col,
                    "timestamp": timestamp_col,
                    "channel": channel_col,
                    "conversion": conversion_col,
                }.items()
                if value is not None
            }
        )
        self.lookback_window = lookback_window
        self.conversion_window = conversion_window
        self.include_non_converters = include_non_converters

    def build(self) -> JourneyCollection:
        """Materialize journeys from the configured event frame."""

        return build_journeys(
            self.data,
            columns=self.columns,
            lookback_window=self.lookback_window,
            conversion_window=self.conversion_window,
            include_non_converters=self.include_non_converters,
        )


def summarize_journeys(journeys: Iterable[Journey]) -> JourneySummary:
    """Compute path-length, frequency, and conversion summary statistics."""

    records = tuple(journeys)
    frequencies: Counter[str] = Counter()
    touch_lengths: list[int] = []
    converter_lengths: list[int] = []
    non_converter_lengths: list[int] = []
    durations: list[pd.Timedelta] = []
    unique_paths: set[tuple[str, ...]] = set()
    repeated_touches = 0
    converter_count = 0
    for journey in records:
        channels = journey.channels
        length = len(channels)
        frequencies.update(channels)
        touch_lengths.append(length)
        unique_paths.add(channels)
        repeated_touches += length - len(set(channels))
        durations.append(journey.duration)
        if journey.converted:
            converter_count += 1
            converter_lengths.append(length)
        else:
            non_converter_lengths.append(length)
    touch_count = int(sum(frequencies.values()))
    n_journeys = len(records)
    return JourneySummary(
        journey_count=n_journeys,
        converter_count=converter_count,
        non_converter_count=n_journeys - converter_count,
        touchpoint_count=touch_count,
        unique_channel_count=len(frequencies),
        average_touches=touch_count / n_journeys if n_journeys else 0.0,
        median_touches=float(pd.Series(touch_lengths).median()) if records else 0.0,
        minimum_touches=min(touch_lengths, default=0),
        maximum_touches=max(touch_lengths, default=0),
        conversion_rate=converter_count / n_journeys if n_journeys else 0.0,
        average_duration=(
            pd.to_timedelta(
                float(
                    np.mean(
                        np.asarray(
                            [duration.value for duration in durations],
                            dtype=np.float64,
                        )
                    )
                ),
                unit="ns",
            )
            if durations
            else pd.Timedelta(0)
        ),
        unique_path_count=len(unique_paths),
        converter_average_path_length=(
            sum(converter_lengths) / len(converter_lengths) if converter_lengths else 0.0
        ),
        non_converter_average_path_length=(
            sum(non_converter_lengths) / len(non_converter_lengths)
            if non_converter_lengths
            else 0.0
        ),
        converter_median_path_length=(
            float(pd.Series(converter_lengths).median()) if converter_lengths else 0.0
        ),
        non_converter_median_path_length=(
            float(pd.Series(non_converter_lengths).median())
            if non_converter_lengths
            else 0.0
        ),
        touchpoint_frequency=frequencies,
        channel_frequency=frequencies,
        repeated_touch_rate=repeated_touches / touch_count if touch_count else 0.0,
    )


def _build_from_canonical(
    canonical: pd.DataFrame,
    *,
    lookback: pd.Timedelta,
    conversion_limit: pd.Timedelta | None,
    include_non_converters: bool,
) -> list[Journey]:
    if canonical.empty:
        return []
    user_ids = canonical["user_id"].to_numpy()
    channels = canonical["channel"].to_numpy()
    timestamps = _as_naive_datetime64(canonical["timestamp"])
    conversions = canonical["conversion"].to_numpy(dtype=bool)
    values = (
        canonical["conversion_value"].to_numpy()
        if "conversion_value" in canonical.columns
        else np.ones(len(canonical), dtype=float)
    )
    conversion_times = (
        _as_naive_datetime64(canonical["conversion_timestamp"])
        if "conversion_timestamp" in canonical.columns
        else None
    )
    change: np.ndarray = np.empty(len(canonical), dtype=bool)
    change[0] = True
    change[1:] = user_ids[1:] != user_ids[:-1]
    starts = np.flatnonzero(change)
    ends: np.ndarray = np.append(starts[1:], len(canonical))
    journeys: list[Journey] = []
    lookback_delta = np.timedelta64(lookback.value, "ns")
    conversion_delta = (
        None if conversion_limit is None else np.timedelta64(conversion_limit.value, "ns")
    )
    for start, end in zip(starts, ends, strict=True):
        group_conv = np.flatnonzero(conversions[start:end])
        user_id = user_ids[start]
        previous = -1
        for local_index in group_conv:
            absolute = start + int(local_index)
            endpoint = timestamps[absolute]
            if conversion_times is not None and not np.isnat(conversion_times[absolute]):
                endpoint = conversion_times[absolute]
            window_start = endpoint - lookback_delta
            if conversion_delta is not None:
                window_start = max(window_start, endpoint - conversion_delta)
            relative = np.arange(end - start)
            include = (
                (relative > previous)
                & (relative <= local_index)
                & (timestamps[start:end] >= window_start)
                & (timestamps[start:end] <= endpoint)
            )
            selected = np.flatnonzero(include)
            if selected.size == 0:
                continue
            selected_abs = start + selected
            journeys.append(
                Journey(
                    user_id=user_id,
                    touchpoints=tuple(
                        Touchpoint(
                            channel=str(channel),
                            timestamp=pd.Timestamp(stamp).tz_localize("UTC"),
                        )
                        for channel, stamp in zip(
                            channels[selected_abs], timestamps[selected_abs], strict=True
                        )
                    ),
                    converted=True,
                    conversion_time=pd.Timestamp(endpoint).tz_localize("UTC"),
                    conversion_value=float(values[absolute]) if pd.notna(values[absolute]) else 1.0,
                )
            )
            previous = int(local_index)
        if include_non_converters and group_conv.size == 0:
            endpoint = timestamps[end - 1]
            window_start = endpoint - lookback_delta
            include = timestamps[start:end] >= window_start
            selected_abs = start + np.flatnonzero(include)
            journeys.append(
                Journey(
                    user_id=user_id,
                    touchpoints=tuple(
                        Touchpoint(
                            channel=str(channel),
                            timestamp=pd.Timestamp(stamp).tz_localize("UTC"),
                        )
                        for channel, stamp in zip(
                            channels[selected_abs], timestamps[selected_abs], strict=True
                        )
                    ),
                    converted=False,
                )
            )
    return journeys


def _positive_timedelta(value: str | timedelta | pd.Timedelta, name: str) -> pd.Timedelta:
    try:
        result = pd.Timedelta(value)
    except (ValueError, TypeError) as exc:
        raise JourneyError(f"{name} must be a valid duration.") from exc
    if result <= pd.Timedelta(0):
        raise JourneyError(f"{name} must be positive.")
    return result


def _as_naive_datetime64(values: pd.Series) -> np.ndarray:
    converted = pd.to_datetime(values, utc=True, errors="coerce")
    naive = converted.dt.tz_convert("UTC").dt.tz_localize(None)
    return naive.to_numpy(dtype="datetime64[ns]")
