import pandas as pd
import pytest

from mta_audit import (
    FirstTouchAttribution,
    LastTouchAttribution,
    LinearAttribution,
    build_journeys,
)


@pytest.fixture
def journeys():
    events = pd.DataFrame(
        {
            "user_id": ["a", "a", "a", "b", "b"],
            "timestamp": pd.date_range("2026-01-01", periods=5, tz="UTC"),
            "channel": ["email", "email", "direct", "social", "direct"],
            "conversion": [0, 0, 1, 0, 1],
            "conversion_value": [0, 0, 30, 0, 10],
        }
    )
    return build_journeys(events)


def test_first_and_last_touch_allocate_value_and_conversions(journeys) -> None:
    first = FirstTouchAttribution().attribute(journeys).data.set_index("channel")
    last = LastTouchAttribution().attribute(journeys).data.set_index("channel")

    assert first.loc["email", "attribution_credit"] == 30
    assert first.loc["social", "conversions"] == 1
    assert last.loc["direct", "attribution_credit"] == 40
    assert last.loc["direct", "share"] == 1


def test_linear_counts_repeated_channels_as_repeated_touches(journeys) -> None:
    result = LinearAttribution().attribute(journeys).data.set_index("channel")

    assert result.loc["email", "attribution_credit"] == pytest.approx(20)
    assert result.loc["direct", "attribution_credit"] == pytest.approx(15)
    assert result["share"].sum() == pytest.approx(1)
    assert result["conversions"].sum() == pytest.approx(2)
