"""Deterministic event-data corruption and duplicate utilities."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import numpy as np
import pandas as pd

from ..schema import ColumnMapping


@dataclass(frozen=True, slots=True)
class CorruptionMetadata(Mapping[str, Any]):
    """Complete record of requested and applied corruptions."""

    requested: Mapping[str, Any]
    applied: Mapping[str, Any]
    affected_users: Mapping[str, tuple[Any, ...]]
    row_counts: Mapping[str, int]
    random_state: int | None

    def __post_init__(self) -> None:
        object.__setattr__(self, "requested", MappingProxyType(dict(self.requested)))
        object.__setattr__(self, "applied", MappingProxyType(dict(self.applied)))
        object.__setattr__(
            self,
            "affected_users",
            MappingProxyType({key: tuple(value) for key, value in self.affected_users.items()}),
        )
        object.__setattr__(self, "row_counts", MappingProxyType(dict(self.row_counts)))

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def __iter__(self) -> Iterator[str]:
        return iter(("requested", "applied", "affected_users", "row_counts", "random_state"))

    def __len__(self) -> int:
        return 5


@dataclass(frozen=True, slots=True)
class CorruptedFrame:
    """A DataFrame-compatible result carrying corruption metadata."""

    data: pd.DataFrame
    metadata: CorruptionMetadata

    def __getattr__(self, name: str) -> Any:
        return getattr(self.data, name)

    def __getitem__(self, key: Any) -> Any:
        return self.data[key]

    def __len__(self) -> int:
        return len(self.data)

    def __iter__(self) -> Iterator[Any]:
        return iter(self.data)

    def __repr__(self) -> str:
        return repr(self.data)

    def to_dataframe(self, *, copy: bool = True) -> pd.DataFrame:
        """Return the underlying frame, copied by default."""

        return self.data.copy(deep=True) if copy else self.data


def corrupt(
    df: pd.DataFrame,
    *,
    identity_loss: float = 0,
    touchpoint_loss: float = 0,
    duplicate_rate: float = 0,
    conversion_delay_shift: object = 0,
    missing_channel_rate: float = 0,
    timestamp_noise: object = 0,
    impression_loss: float = 0,
    click_loss: float = 0,
    columns: ColumnMapping | dict[str, str] | None = None,
    user_col: str | None = None,
    timestamp_col: str | None = None,
    channel_col: str | None = None,
    conversion_col: str | None = None,
    event_type_col: str | None = None,
    duplicate_kind: str = "exact",
    random_state: int | None = 42,
) -> CorruptedFrame:
    """Return a corrupted copy and an exact, deterministic application record.

    Identity loss fragments selected users by moving a deterministic sample of their
    non-conversion events to synthetic IDs. Generic and event-specific event loss
    never removes conversion rows.
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")
    rates = {
        "identity_loss": identity_loss,
        "touchpoint_loss": touchpoint_loss,
        "duplicate_rate": duplicate_rate,
        "missing_channel_rate": missing_channel_rate,
        "impression_loss": impression_loss,
        "click_loss": click_loss,
    }
    for name, value in rates.items():
        _validate_rate(value, name)
    mapping = _resolve_mapping(
        columns, user_col, timestamp_col, channel_col, conversion_col
    )
    impression_col = mapping.impression or ("impression" if "impression" in df else None)
    click_col = mapping.click or ("click" if "click" in df else None)
    required = {mapping.user_id, mapping.timestamp, mapping.channel, mapping.conversion}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if event_type_col is not None and event_type_col not in df.columns:
        raise ValueError("event_type_col must identify an existing column for event loss.")
    if impression_loss and event_type_col is None and not impression_col:
        raise ValueError("Map an impression column or provide event_type_col for event loss.")
    if click_loss and event_type_col is None and not click_col:
        raise ValueError("Map a click column or provide event_type_col for event loss.")
    if duplicate_kind not in {"exact", "near"}:
        raise ValueError("duplicate_kind must be 'exact' or 'near'.")

    requested_delay = conversion_delay_shift
    requested_noise = timestamp_noise
    delay = _duration(conversion_delay_shift, "conversion_delay_shift", signed=True)
    noise = _duration(timestamp_noise, "timestamp_noise")
    rng = np.random.default_rng(random_state)
    result = df.copy(deep=True)
    original_rows = len(result)
    affected: dict[str, tuple[Any, ...]] = {}
    counts: dict[str, int] = {"input": original_rows, "initial_rows": original_rows}
    applied: dict[str, Any] = {}

    conversion = _conversion_mask(result[mapping.conversion])

    # Fragment a rate of eligible user journeys, keeping conversion ownership intact.
    non_conversion_counts = result.loc[~conversion].groupby(mapping.user_id).size()
    converting_users = set(result.loc[conversion, mapping.user_id])
    eligible_users = np.asarray(
        [
            user
            for user, count in non_conversion_counts.items()
            if count >= 2 or user in converting_users
        ],
        dtype=object,
    )
    selected_users = _sample_values(eligible_users, identity_loss, rng)
    changed_identity_indices: list[Any] = []
    if len(selected_users):
        result[mapping.user_id] = result[mapping.user_id].astype(object)
    for ordinal, user in enumerate(selected_users):
        candidates = result.index[(result[mapping.user_id] == user) & ~conversion].tolist()
        if not candidates:
            continue
        # Move roughly half the touches so the original and synthetic IDs both remain
        # where possible; a one-touch pre-conversion path is necessarily fully moved.
        move_count = max(1, len(candidates) // 2)
        chosen = _sample_indices(candidates, move_count, rng)
        synthetic = f"__mta_fragment__{ordinal}__{user}"
        result.loc[chosen, mapping.user_id] = synthetic
        changed_identity_indices.extend(chosen)
    affected["identity_loss"] = tuple(selected_users.tolist())
    counts["identity_rows_changed"] = len(changed_identity_indices)
    applied["identity_loss"] = (
        len(selected_users) / len(eligible_users) if len(eligible_users) else 0.0
    )

    # Losses are applied to current row identities but eligibility is based on event
    # semantics, and conversions are always protected.
    drop_by_type: dict[str, list[Any]] = {}
    available = result.index[~conversion].tolist()
    drop_by_type["touchpoint_loss"] = _sample_rate_indices(available, touchpoint_loss, rng)
    reserved = set(drop_by_type["touchpoint_loss"])
    type_masks: dict[str, pd.Series] = {}
    if event_type_col:
        normalized_types = result[event_type_col].astype(str).str.strip().str.lower()
        type_masks = {
            "impression": normalized_types.eq("impression"),
            "click": normalized_types.eq("click"),
        }
    else:
        if impression_col:
            type_masks["impression"] = _conversion_mask(result[impression_col])
        if click_col:
            type_masks["click"] = _conversion_mask(result[click_col])
    if type_masks:
        for name, label, rate in (
            ("impression_loss", "impression", impression_loss),
            ("click_loss", "click", click_loss),
        ):
            event_mask = type_masks.get(label, pd.Series(False, index=result.index))
            eligible = result.index[
                (~conversion) & event_mask & ~result.index.isin(reserved)
            ].tolist()
            chosen = _sample_rate_indices(eligible, rate, rng)
            drop_by_type[name] = chosen
            reserved.update(chosen)
    else:
        drop_by_type.update({"impression_loss": [], "click_loss": []})
    drop_indices = [idx for indices in drop_by_type.values() for idx in indices]
    for name, indices in drop_by_type.items():
        users = pd.unique(result.loc[indices, mapping.user_id]).tolist() if indices else []
        affected[name] = tuple(users)
        counts[f"{name}_rows_removed"] = len(indices)
        eligible_count = (
            len(available)
            if name == "touchpoint_loss"
            else int(
                (
                    ~conversion
                    & type_masks.get(
                        name.removesuffix("_loss"), pd.Series(False, index=result.index)
                    )
                ).sum()
            )
            if type_masks
            else 0
        )
        applied[name] = len(indices) / eligible_count if eligible_count else 0.0
    if drop_indices:
        result = result.drop(index=drop_indices)

    current_conversion = _conversion_mask(result[mapping.conversion])
    if delay:
        result.loc[current_conversion, mapping.timestamp] = (
            pd.to_datetime(result.loc[current_conversion, mapping.timestamp])
            + delay
        )
    delay_count = int(current_conversion.sum()) if delay else 0
    affected["conversion_delay_shift"] = tuple(
        pd.unique(result.loc[current_conversion, mapping.user_id]).tolist() if delay else []
    )
    counts["conversion_timestamps_shifted"] = delay_count
    applied["conversion_delay_shift"] = delay

    if missing_channel_rate:
        eligible = result.index.tolist()
        missing_indices = _sample_rate_indices(eligible, missing_channel_rate, rng)
        result.loc[missing_indices, mapping.channel] = pd.NA
    else:
        missing_indices = []
    affected["missing_channel_rate"] = tuple(
        pd.unique(result.loc[missing_indices, mapping.user_id]).tolist()
        if missing_indices
        else []
    )
    counts["channels_missing"] = len(missing_indices)
    applied["missing_channel_rate"] = len(missing_indices) / len(result) if len(result) else 0.0

    if noise:
        noise_indices = result.index.tolist()
        nanoseconds = int(noise.value)
        offsets = rng.integers(-nanoseconds, nanoseconds + 1, size=len(noise_indices))
        timestamps = pd.to_datetime(result[mapping.timestamp])
        timestamps = pd.Series(timestamps.array.as_unit("ns"), index=result.index)
        result[mapping.timestamp] = (
            timestamps + pd.to_timedelta(offsets, unit="ns")
        )
    else:
        noise_indices = []
    affected["timestamp_noise"] = tuple(
        pd.unique(result.loc[noise_indices, mapping.user_id]).tolist()
        if noise_indices
        else []
    )
    counts["timestamps_noised"] = len(noise_indices)
    applied["timestamp_noise"] = noise

    duplicate_count = _rate_count(len(result), duplicate_rate)
    if duplicate_count:
        chosen = _sample_indices(result.index.tolist(), duplicate_count, rng)
        copies = result.loc[chosen].copy(deep=True)
        if duplicate_kind == "near":
            copies[mapping.timestamp] = (
                pd.to_datetime(copies[mapping.timestamp]) + pd.Timedelta(microseconds=1)
            )
        result = pd.concat([result, copies], ignore_index=True)
        duplicate_users = pd.unique(copies[mapping.user_id]).tolist()
    else:
        duplicate_users = []
    affected["duplicate_rate"] = tuple(duplicate_users)
    counts["duplicate_rows_added"] = duplicate_count
    applied["duplicate_rate"] = (
        duplicate_count / (len(result) - duplicate_count) if len(result) else 0
    )
    counts["output"] = len(result)
    counts["final_rows"] = len(result)
    counts["net_change"] = len(result) - original_rows

    requested = {
        **rates,
        "conversion_delay_shift": requested_delay,
        "timestamp_noise": requested_noise,
        "duplicate_kind": duplicate_kind,
    }
    metadata = CorruptionMetadata(requested, applied, affected, counts, random_state)
    return CorruptedFrame(result.reset_index(drop=True), metadata)


def detect_duplicates(
    df: pd.DataFrame,
    *,
    columns: ColumnMapping | dict[str, str] | None = None,
    event_type_col: str | None = None,
    near_tolerance: object = "1s",
) -> pd.DataFrame:
    """Classify exact, near, and repeated semantic events."""

    mapping = ColumnMapping.from_value(columns)
    required = [mapping.user_id, mapping.timestamp, mapping.channel, mapping.conversion]
    missing = [column for column in required if column not in df]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    work = df.copy()
    work["_exact"] = work.duplicated(keep=False)
    work["_position"] = np.arange(len(work))
    work["_timestamp"] = pd.to_datetime(work[mapping.timestamp])
    semantic = [mapping.user_id, mapping.channel, mapping.conversion]
    if event_type_col and event_type_col in work:
        semantic.append(event_type_col)
    ordered = work.sort_values([*semantic, "_timestamp"], kind="stable")
    gaps = ordered.groupby(semantic, dropna=False)["_timestamp"].diff().abs()
    tolerance = _duration(near_tolerance, "near_tolerance")
    ordered["_near"] = gaps.le(tolerance) & ~ordered["_exact"]
    ordered["_repeated"] = ordered.duplicated(subset=semantic, keep=False)
    classified = ordered.set_index("_position").sort_index()
    kind = np.select(
        [classified["_exact"], classified["_near"], classified["_repeated"]],
        ["exact", "near", "repeated"],
        default="unique",
    )
    return pd.DataFrame(
        {
            "kind": kind,
            "is_duplicate": kind != "unique",
            "is_exact": classified["_exact"].to_numpy(),
            "is_near": classified["_near"].to_numpy(),
            "is_repeated": classified["_repeated"].to_numpy(),
        },
        index=df.index,
    )


def deduplicate(
    df: pd.DataFrame,
    *,
    columns: ColumnMapping | dict[str, str] | None = None,
    event_type_col: str | None = None,
    near_tolerance: object | None = None,
) -> pd.DataFrame:
    """Remove later exact duplicates and, optionally, near duplicates."""

    result = df.drop_duplicates(keep="first").copy()
    if near_tolerance is None or result.empty:
        return result.reset_index(drop=True)
    mapping = ColumnMapping.from_value(columns)
    semantic = [mapping.user_id, mapping.channel, mapping.conversion]
    if event_type_col and event_type_col in result:
        semantic.append(event_type_col)
    tolerance = _duration(near_tolerance, "near_tolerance")
    timestamps = pd.to_datetime(result[mapping.timestamp])
    ordered = result.assign(_timestamp=timestamps).sort_values(
        [*semantic, "_timestamp"], kind="stable"
    )
    gap = ordered.groupby(semantic, dropna=False)["_timestamp"].diff()
    return ordered.loc[gap.isna() | gap.gt(tolerance)].drop(columns="_timestamp").reset_index(
        drop=True
    )


def _resolve_mapping(
    columns: ColumnMapping | dict[str, str] | None,
    user_col: str | None,
    timestamp_col: str | None,
    channel_col: str | None,
    conversion_col: str | None,
) -> ColumnMapping:
    direct = {
        key: value
        for key, value in {
            "user_id": user_col,
            "timestamp": timestamp_col,
            "channel": channel_col,
            "conversion": conversion_col,
        }.items()
        if value is not None
    }
    if columns is not None and direct:
        raise ValueError("Use either columns or individual *_col arguments, not both.")
    return ColumnMapping.from_value(direct or columns)


def _validate_rate(value: float, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value <= 1:
        raise ValueError(f"{name} must be between 0 and 1.")


def _rate_count(size: int, rate: float) -> int:
    if not size or not rate:
        return 0
    return min(size, max(1, round(size * rate)))


def _sample_indices(values: list[Any], count: int, rng: np.random.Generator) -> list[Any]:
    if not count:
        return []
    positions = np.sort(rng.choice(len(values), size=count, replace=False))
    return [values[int(position)] for position in positions]


def _sample_rate_indices(
    values: list[Any], rate: float, rng: np.random.Generator
) -> list[Any]:
    return _sample_indices(values, _rate_count(len(values), rate), rng)


def _sample_values(
    values: np.ndarray, rate: float, rng: np.random.Generator
) -> np.ndarray:
    count = _rate_count(len(values), rate)
    if not count:
        return values[:0]
    return values[np.sort(rng.choice(len(values), size=count, replace=False))]


def _conversion_mask(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    normalized = series.astype(str).str.strip().str.lower()
    return normalized.isin({"1", "true", "t", "yes", "y"})


def _duration(value: object, name: str, *, signed: bool = False) -> pd.Timedelta:
    if value in (0, None, ""):
        return pd.Timedelta(0)
    try:
        duration = pd.Timedelta(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be zero or a valid duration.") from exc
    if not signed and duration < pd.Timedelta(0):
        raise ValueError(f"{name} cannot be negative.")
    return duration
