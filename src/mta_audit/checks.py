"""Phase 1 event data-quality and duplicate checks."""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

from .results import AuditFinding, AuditResult, AuditScore, Severity
from .schema import ColumnMapping

_PENALTIES = {
    Severity.INFO: 0.0,
    Severity.LOW: 2.0,
    Severity.MEDIUM: 7.0,
    Severity.HIGH: 15.0,
    Severity.CRITICAL: 30.0,
}


def run_data_quality_checks(
    events: pd.DataFrame,
    columns: ColumnMapping | dict[str, str] | None = None,
    *,
    reference_time: object | None = None,
    reference_timestamp: object | None = None,
    future_tolerance: object = "0s",
    minimum_timestamp: object | None = "1970-01-01",
) -> AuditResult:
    """Check schema, timestamps, identifiers, and journey plausibility.

    ``reference_time`` controls future-event semantics (the current UTC time by
    default). ``reference_timestamp`` is a compatibility alias.
    """

    if not isinstance(events, pd.DataFrame):
        raise TypeError("events must be a pandas DataFrame")
    mapping = ColumnMapping.from_value(columns)
    findings: list[AuditFinding] = []
    required = {
        "user_id": mapping.user_id,
        "timestamp": mapping.timestamp,
        "channel": mapping.channel,
        "conversion": mapping.conversion,
    }
    missing = [canonical for canonical, source in required.items() if source not in events]
    if missing:
        findings.append(
            AuditFinding(
                code="missing_columns",
                severity=Severity.CRITICAL,
                message=f"Required canonical fields are not mapped to data: {missing}",
                count=len(missing),
                details={"fields": missing},
            )
        )
        return _result("data_quality", findings)

    for canonical, source in required.items():
        null_count = int(events[source].isna().sum())
        if null_count:
            severity = Severity.HIGH if canonical != "conversion" else Severity.MEDIUM
            findings.append(
                AuditFinding(
                    code=f"null_{canonical}",
                    severity=severity,
                    message=f"{canonical} contains {null_count} null values.",
                    count=null_count,
                )
            )

    non_null_timestamps = events[mapping.timestamp].dropna()
    parsed = pd.to_datetime(non_null_timestamps, errors="coerce", utc=True, format="mixed")
    invalid_timestamp_count = int(parsed.isna().sum())
    if invalid_timestamp_count:
        findings.append(
            AuditFinding(
                code="invalid_timestamp",
                severity=Severity.HIGH,
                message=f"{invalid_timestamp_count} timestamps cannot be parsed.",
                count=invalid_timestamp_count,
            )
        )
    valid_timestamps = parsed.dropna()
    reference = reference_timestamp if reference_timestamp is not None else reference_time
    reference_value = pd.Timestamp.now(tz="UTC") if reference is None else pd.Timestamp(reference)
    if reference_value.tzinfo is None:
        reference_value = reference_value.tz_localize("UTC")
    else:
        reference_value = reference_value.tz_convert("UTC")
    tolerance = pd.Timedelta(future_tolerance)
    future_count = int((valid_timestamps > reference_value + tolerance).sum())
    if future_count:
        findings.append(
            _metric_finding(
                "future_timestamps",
                Severity.HIGH,
                future_count,
                len(events),
                events,
                mapping,
                "timestamp",
                f"{future_count} timestamps occur after the configured reference time.",
            )
        )
    if minimum_timestamp is not None:
        minimum = pd.Timestamp(minimum_timestamp)
        minimum = (
            minimum.tz_localize("UTC") if minimum.tzinfo is None else minimum.tz_convert("UTC")
        )
        impossible_count = int((valid_timestamps < minimum).sum())
        if impossible_count:
            findings.append(
                _metric_finding(
                    "impossible_timestamps",
                    Severity.HIGH,
                    impossible_count,
                    len(events),
                    events,
                    mapping,
                    "timestamp",
                    f"{impossible_count} timestamps precede the configured minimum.",
                )
            )

    empty_channels = int(
        events[mapping.channel].dropna().astype(str).str.strip().eq("").sum()
    )
    if empty_channels:
        findings.append(
            AuditFinding(
                code="empty_channel",
                severity=Severity.HIGH,
                message=f"{empty_channels} channel values are empty.",
                count=empty_channels,
            )
        )
    conversion_mask = _truthy(events[mapping.conversion])
    conversion_timestamp = mapping.conversion_timestamp or (
        "conversion_timestamp" if "conversion_timestamp" in events else None
    )
    if conversion_timestamp:
        conversion_times = pd.to_datetime(
            events[conversion_timestamp], errors="coerce", utc=True, format="mixed"
        )
        invalid_conversion_times = int(
            (events[conversion_timestamp].notna() & conversion_times.isna()).sum()
        )
        if invalid_conversion_times:
            findings.append(
                AuditFinding(
                    code="invalid_conversion_timestamp",
                    severity=Severity.HIGH,
                    message=(
                        f"{invalid_conversion_times} conversion timestamps cannot be parsed."
                    ),
                    count=invalid_conversion_times,
                )
            )
        exposure_times = pd.to_datetime(
            events[mapping.timestamp], errors="coerce", utc=True, format="mixed"
        )
        first_exposure = exposure_times.where(~conversion_mask).groupby(
            events[mapping.user_id], dropna=False
        ).transform("min")
        before = conversion_mask & conversion_times.notna() & (
            first_exposure.isna() | conversion_times.lt(first_exposure)
        )
        count = int(before.sum())
        if count:
            findings.append(
                _metric_finding(
                    "conversion_before_exposure",
                    Severity.HIGH,
                    count,
                    int(conversion_mask.sum()),
                    events.loc[before],
                    mapping,
                    "conversion",
                    f"{count} conversions occur before any observed exposure.",
                )
            )
    exposure_users = set(events.loc[~conversion_mask, mapping.user_id].dropna())
    malformed = conversion_mask & ~events[mapping.user_id].isin(exposure_users)
    malformed_count = int(malformed.sum())
    if malformed_count:
        findings.append(
            _metric_finding(
                "malformed_journeys",
                Severity.MEDIUM,
                malformed_count,
                int(conversion_mask.sum()),
                events.loc[malformed],
                mapping,
                "conversion",
                f"{malformed_count} conversions have no separate exposure event.",
            )
        )
    conversion_ids = tuple(
        dict.fromkeys(
            source
            for source in (
                mapping.conversion_id,
                "conversion_id" if "conversion_id" in events else None,
                mapping.order_id,
                "order_id" if "order_id" in events else None,
            )
            if source
        )
    )
    repeated_ids = pd.Series(False, index=events.index)
    for conversion_id in conversion_ids:
        repeated_ids |= (
            conversion_mask
            & events[conversion_id].notna()
            & events[conversion_id].duplicated(keep=False)
        )
    repeated_count = int(repeated_ids.sum())
    if repeated_count:
        findings.append(
            _metric_finding(
                "repeated_conversion_ids",
                Severity.HIGH,
                repeated_count,
                int(conversion_mask.sum()),
                events.loc[repeated_ids],
                mapping,
                "conversion",
                f"{repeated_count} conversion rows reuse an order/conversion ID.",
            )
        )
    value_col = mapping.conversion_value or (
        "conversion_value" if "conversion_value" in events else None
    )
    if value_col and conversion_mask.any():
        values = pd.to_numeric(events.loc[conversion_mask, value_col], errors="coerce")
        if float(values.fillna(0).sum()) == 0:
            findings.append(
                AuditFinding(
                    code="zero_conversion_value",
                    severity=Severity.CRITICAL,
                    message=(
                        "Converting events have missing or zero conversion_value; "
                        "channel shares are not decision-grade."
                    ),
                    count=int(conversion_mask.sum()),
                    metric_name="conversion_value_sum",
                    metric_value=0.0,
                    recommendation=(
                        "Supply order revenue/value before using attribution shares."
                    ),
                )
            )
    return _result("data_quality", findings)


def run_duplicate_checks(
    events: pd.DataFrame,
    columns: ColumnMapping | dict[str, str] | None = None,
) -> AuditResult:
    """Find exact rows and duplicate event identities."""

    if not isinstance(events, pd.DataFrame):
        raise TypeError("events must be a pandas DataFrame")
    mapping = ColumnMapping.from_value(columns)
    findings: list[AuditFinding] = []
    parsed_for_grain = (
        pd.to_datetime(events[mapping.timestamp], errors="coerce", utc=True, format="mixed")
        if mapping.timestamp in events
        else pd.Series(dtype="datetime64[ns, UTC]")
    )
    date_grain = _is_date_grain(parsed_for_grain)
    exact_mask = events.duplicated(keep="first")
    exact_count = int(exact_mask.sum())
    identity = [mapping.user_id, mapping.timestamp, mapping.channel]
    has_identity = all(column in events for column in identity)
    business_mask = (
        events.duplicated(subset=identity, keep="first")
        if has_identity
        else pd.Series(False, index=events.index)
    )
    business_count = int(business_mask.sum())
    date_grain_repeats = max(exact_count, business_count)
    if date_grain and date_grain_repeats:
        findings.append(
            AuditFinding(
                code="date_grain_repeated_touches",
                severity=Severity.INFO,
                message=(
                    f"{date_grain_repeats} repeated user/channel/day rows look like "
                    "date-grain touches, not injected duplicates."
                ),
                count=date_grain_repeats,
                metric_name="date_grain_repeat_rate",
                metric_value=date_grain_repeats / len(events) if len(events) else 0.0,
                recommendation=(
                    "Use second-level timestamps before treating same-day repeats as duplicates."
                ),
            )
        )
    elif exact_count:
        findings.append(
            _metric_finding(
                "exact_duplicates",
                Severity.MEDIUM,
                exact_count,
                len(events),
                events.loc[exact_mask],
                mapping,
                "event",
                f"{exact_count} rows exactly duplicate an earlier row.",
            )
        )

    if has_identity:
        non_exact_count = max(0, business_count - exact_count)
        if non_exact_count and not date_grain:
            findings.append(
                AuditFinding(
                    code="event_identity_duplicates",
                    severity=Severity.MEDIUM,
                    message=(
                        f"{non_exact_count} rows reuse user, timestamp, and channel "
                        "with differing attributes."
                    ),
                    count=non_exact_count,
                    metric_name="duplicate_event_rate",
                    metric_value=non_exact_count / len(events) if len(events) else 0.0,
                    details=_affected_metrics(
                        events.loc[business_mask & ~exact_mask],
                        mapping,
                        non_exact_count,
                        len(events),
                    ),
                )
            )
        parsed = pd.to_datetime(
            events[mapping.timestamp], errors="coerce", utc=True, format="mixed"
        )
        ordered = events.assign(_parsed_timestamp=parsed, _position=range(len(events))).sort_values(
            [mapping.user_id, mapping.channel, "_parsed_timestamp"], kind="stable"
        )
        gaps = ordered.groupby([mapping.user_id, mapping.channel], dropna=False)[
            "_parsed_timestamp"
        ].diff()
        near_positions = ordered.loc[
            gaps.between(pd.Timedelta(0), pd.Timedelta("1s"), inclusive="right")
            & ~ordered.index.isin(events.index[exact_mask]),
            "_position",
        ]
        near_count = len(near_positions)
        if near_count and not date_grain:
            near_rows = events.iloc[near_positions.to_numpy()]
            findings.append(
                _metric_finding(
                    "near_duplicates",
                    Severity.MEDIUM,
                    near_count,
                    len(events),
                    near_rows,
                    mapping,
                    "event",
                    f"{near_count} events are near duplicates within one second.",
                )
            )

    conversion_mask = _truthy(events[mapping.conversion]) if mapping.conversion in events else None
    indicators = {
        "impressions": mapping.impression or ("impression" if "impression" in events else None),
        "clicks": mapping.click or ("click" if "click" in events else None),
        "conversions": mapping.conversion,
    }
    for label, source in indicators.items():
        if source is None or source not in events:
            continue
        active = conversion_mask if label == "conversions" else _truthy(events[source])
        semantic = [column for column in identity if column in events]
        repeated = active & events.duplicated(subset=semantic, keep="first") & ~exact_mask
        count = int(repeated.sum())
        if count and not date_grain:
            findings.append(
                _metric_finding(
                    f"repeated_{label}",
                    Severity.MEDIUM,
                    count,
                    int(active.sum()),
                    events.loc[repeated],
                    mapping,
                    label[:-1],
                    f"{count} {label} repeat an earlier semantic event.",
                )
            )
    return _result("duplicates", findings)


def _is_date_grain(parsed: pd.Series) -> bool:
    valid = parsed.dropna()
    if valid.empty:
        return False
    midnight = (
        (valid.dt.hour == 0)
        & (valid.dt.minute == 0)
        & (valid.dt.second == 0)
        & (valid.dt.microsecond == 0)
    )
    return bool(midnight.mean() >= 0.9)


def _truthy(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    return series.astype(str).str.strip().str.lower().isin({"1", "true", "t", "yes", "y"})


def _affected_metrics(
    rows: pd.DataFrame,
    mapping: ColumnMapping,
    count: int,
    denominator: int,
) -> dict[str, float | int]:
    conversions = (
        int(_truthy(rows[mapping.conversion]).sum()) if mapping.conversion in rows else 0
    )
    return {
        "duplicate_event_rate": count / denominator if denominator else 0.0,
        "affected_users": (
            int(rows[mapping.user_id].nunique(dropna=True)) if mapping.user_id in rows else 0
        ),
        "affected_conversions": conversions,
    }


def _metric_finding(
    code: str,
    severity: Severity,
    count: int,
    denominator: int,
    rows: pd.DataFrame,
    mapping: ColumnMapping,
    unit: str,
    message: str,
) -> AuditFinding:
    rate = count / denominator if denominator else 0.0
    details = _affected_metrics(rows, mapping, count, denominator)
    details["unit"] = unit
    return AuditFinding(
        code=code,
        severity=severity,
        message=message,
        count=count,
        metric_name="duplicate_event_rate" if "duplicate" in code else f"{code}_rate",
        metric_value=rate,
        details=details,
    )


def combine_scores(results: Iterable[AuditResult]) -> AuditScore:
    """Combine independent check deductions into one score."""

    deductions: dict[str, float] = {}
    for result in results:
        for code, amount in result.score.deductions.items():
            deductions[f"{result.name}.{code}"] = amount
    return AuditScore(max(0.0, 100.0 - sum(deductions.values())), deductions)


def _result(name: str, findings: list[AuditFinding]) -> AuditResult:
    deductions = {
        finding.code: min(30.0, _PENALTIES[finding.severity] + min(finding.count, 100) / 10)
        for finding in findings
    }
    return AuditResult(
        name=name,
        findings=tuple(findings),
        score=AuditScore(max(0.0, 100.0 - sum(deductions.values())), deductions),
    )
