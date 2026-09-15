import pandas as pd
import pytest

from mta_audit import ColumnMapping, SchemaError, validate_events


def test_mapped_columns_are_normalized_without_mutating_input() -> None:
    source = pd.DataFrame(
        {
            "person": ["a", "a"],
            "occurred": ["2026-01-01", "2026-01-02"],
            "source": ["search", "email"],
            "sale": [0, 1],
            "revenue": [0, 25],
        }
    )
    original = source.copy()

    normalized, summary = validate_events(
        source,
        {
            "user_id": "person",
            "timestamp": "occurred",
            "channel": "source",
            "conversion": "sale",
            "conversion_value": "revenue",
        },
    )

    assert normalized.columns.tolist() == [
        "user_id",
        "timestamp",
        "channel",
        "conversion",
        "conversion_value",
    ]
    assert summary.conversion_count == 1
    pd.testing.assert_frame_equal(source, original)


def test_invalid_schema_values_raise_clear_errors() -> None:
    events = pd.DataFrame(
        {
            "user_id": ["a"],
            "timestamp": ["not-a-date"],
            "channel": ["email"],
            "conversion": [False],
        }
    )
    with pytest.raises(SchemaError, match="timestamp"):
        validate_events(events)

    with pytest.raises(SchemaError, match="distinct"):
        ColumnMapping(user_id="id", timestamp="id")
