"""Input schema configuration and validation."""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any

import pandas as pd


class SchemaError(ValueError):
    """Raised when event data does not satisfy the configured schema."""


@dataclass(frozen=True, slots=True)
class ColumnMapping:
    """Map canonical event fields to caller-owned DataFrame column names."""

    user_id: str = "user_id"
    timestamp: str = "timestamp"
    channel: str = "channel"
    conversion: str = "conversion"
    conversion_value: str | None = None
    conversion_timestamp: str | None = None
    campaign_id: str | None = None
    campaign: str | None = None
    impression: str | None = None
    click: str | None = None
    cost: str | None = None
    revenue: str | None = None
    device: str | None = None
    publisher: str | None = None
    creative_id: str | None = None
    conversion_id: str | None = None
    order_id: str | None = None

    def __post_init__(self) -> None:
        if self.campaign and self.campaign_id and self.campaign != self.campaign_id:
            raise SchemaError("campaign and campaign_id aliases must map to the same column.")
        configured = [value for value in self.as_dict().values() if value is not None]
        if self.campaign and self.campaign_id:
            configured.remove(self.campaign)
        if any(not isinstance(value, str) or not value.strip() for value in configured):
            raise SchemaError("Mapped column names must be non-empty strings.")
        if len(configured) != len(set(configured)):
            raise SchemaError("Each canonical field must map to a distinct column.")

    def as_dict(self) -> dict[str, str | None]:
        """Return canonical names mapped to source names."""

        return {field.name: getattr(self, field.name) for field in fields(self)}

    @classmethod
    def from_value(cls, value: ColumnMapping | dict[str, str] | None) -> ColumnMapping:
        """Normalize a mapping accepted by the public API."""

        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        if isinstance(value, dict):
            unknown = set(value) - {field.name for field in fields(cls)}
            if unknown:
                raise SchemaError(f"Unknown mapped fields: {sorted(unknown)}")
            return cls(**value)
        raise TypeError("columns must be a ColumnMapping, dict, or None")


@dataclass(frozen=True, slots=True)
class ValidationSummary:
    """Summary of schema validation and normalization."""

    row_count: int
    user_count: int
    conversion_count: int


def validate_events(
    events: pd.DataFrame,
    columns: ColumnMapping | dict[str, str] | None = None,
) -> tuple[pd.DataFrame, ValidationSummary]:
    """Validate event data and return a canonical, safely copied DataFrame.

    Source columns are selected through ``columns`` and renamed to canonical names.
    Timestamps are converted with pandas; invalid values fail rather than disappear.
    Conversion values accept booleans, 0/1 numbers, and common true/false strings.
    """

    if not isinstance(events, pd.DataFrame):
        raise TypeError("events must be a pandas DataFrame")
    mapping = ColumnMapping.from_value(columns)
    required = {
        mapping.user_id,
        mapping.timestamp,
        mapping.channel,
        mapping.conversion,
    }
    missing = sorted(column for column in required if column not in events.columns)
    if missing:
        raise SchemaError(f"Missing required columns: {missing}")
    optional = {
        canonical: source
        for canonical, source in mapping.as_dict().items()
        if canonical not in {"user_id", "timestamp", "channel", "conversion", "campaign"}
        and source is not None
    }
    missing_optional = sorted(
        source for source in optional.values() if source not in events.columns
    )
    if missing_optional:
        raise SchemaError(f"Missing configured optional columns: {missing_optional}")
    campaign_source = mapping.campaign_id or mapping.campaign
    if campaign_source and campaign_source not in events.columns:
        raise SchemaError(f"Missing campaign column: {campaign_source!r}")
    mapped_fields = mapping.as_dict()
    if mapping.conversion_value is None and "conversion_value" in events.columns:
        mapped_fields["conversion_value"] = "conversion_value"
    if campaign_source:
        mapped_fields["campaign_id"] = campaign_source
        mapped_fields["campaign"] = None
    configured_sources = {value for value in mapped_fields.values() if value is not None}
    if (
        mapping.campaign is None
        and "campaign" in events.columns
        and "campaign" not in configured_sources
    ):
        mapped_fields["campaign"] = "campaign"
    selected = {source: canonical for canonical, source in mapped_fields.items() if source}
    # Preserve unmapped columns: diagnostics and corruption routines often need
    # event attributes beyond the attribution core.
    canonical = events.copy(deep=True).rename(columns=selected)
    if "campaign_id" in canonical and "campaign" not in canonical:
        canonical["campaign"] = canonical["campaign_id"]
    if canonical.empty:
        canonical["timestamp"] = pd.to_datetime(canonical["timestamp"])
        canonical["conversion"] = canonical["conversion"].astype(bool)
        return canonical, ValidationSummary(0, 0, 0)

    null_key_counts = canonical[["user_id", "timestamp", "channel"]].isna().sum()
    if null_key_counts.any():
        details = ", ".join(f"{key}={count}" for key, count in null_key_counts.items() if count)
        raise SchemaError(f"Required fields contain null values: {details}")

    try:
        canonical["timestamp"] = pd.to_datetime(canonical["timestamp"], errors="raise", utc=True)
    except (ValueError, TypeError) as exc:
        raise SchemaError("timestamp contains values that cannot be parsed.") from exc
    if "conversion_timestamp" in canonical:
        try:
            canonical["conversion_timestamp"] = pd.to_datetime(
                canonical["conversion_timestamp"], errors="raise", utc=True
            )
        except (ValueError, TypeError) as exc:
            raise SchemaError(
                "conversion_timestamp contains values that cannot be parsed."
            ) from exc
    canonical["conversion"] = canonical["conversion"].map(_coerce_conversion)
    canonical["channel"] = canonical["channel"].astype(str).str.strip()
    if canonical["channel"].eq("").any():
        raise SchemaError("channel contains empty values.")
    if "conversion_value" in canonical:
        canonical["conversion_value"] = pd.to_numeric(
            canonical["conversion_value"], errors="coerce"
        )
        invalid_values = canonical["conversion"] & canonical["conversion_value"].isna()
        if invalid_values.any():
            raise SchemaError("Converted rows require numeric conversion_value values.")

    canonical["_user_order"] = pd.factorize(canonical["user_id"], sort=False)[0]
    canonical = canonical.sort_values(
        ["_user_order", "timestamp"], kind="stable", ignore_index=True
    ).drop(columns="_user_order")
    summary = ValidationSummary(
        row_count=len(canonical),
        user_count=int(canonical["user_id"].nunique()),
        conversion_count=int(canonical["conversion"].sum()),
    )
    return canonical, summary


def _coerce_conversion(value: Any) -> bool:
    if isinstance(value, (bool, int, float)) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "t", "yes", "y", "1"}:
            return True
        if normalized in {"false", "f", "no", "n", "0"}:
            return False
    raise SchemaError(f"Invalid conversion indicator: {value!r}")
