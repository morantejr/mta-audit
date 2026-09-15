import pandas as pd
import pytest

from mta_audit import JourneyError, build_journeys


def test_builds_converter_and_non_converter_paths_with_repeats_and_direct() -> None:
    events = pd.DataFrame(
        {
            "uid": ["buyer", "buyer", "buyer", "browser", "browser"],
            "at": [
                "2026-01-01",
                "2026-01-03",
                "2026-01-05",
                "2026-01-02",
                "2026-01-04",
            ],
            "medium": ["email", "email", "direct", "social", "direct"],
            "converted": [0, 0, 1, 0, 0],
        }
    )

    built = build_journeys(
        events,
        columns={
            "user_id": "uid",
            "timestamp": "at",
            "channel": "medium",
            "conversion": "converted",
        },
    )

    assert built.summary.journey_count == 2
    assert built.summary.converter_count == 1
    assert built.summary.non_converter_count == 1
    assert built.journeys[0].channels == ("email", "email", "direct")
    assert built.journeys[1].channels == ("social", "direct")


def test_windows_trim_paths_and_multiple_conversions_do_not_reuse_touches() -> None:
    events = pd.DataFrame(
        {
            "user_id": ["u"] * 5,
            "timestamp": pd.to_datetime(
                ["2026-01-01", "2026-01-09", "2026-01-10", "2026-01-11", "2026-01-12"],
                utc=True,
            ),
            "channel": ["old", "search", "email", "social", "direct"],
            "conversion": [0, 0, 1, 0, 1],
        }
    )

    built = build_journeys(events, lookback_window="30D", conversion_window="3D")

    assert [journey.channels for journey in built] == [
        ("search", "email"),
        ("social", "direct"),
    ]
    assert all(journey.converted for journey in built)


def test_invalid_window_is_rejected() -> None:
    events = pd.DataFrame(columns=["user_id", "timestamp", "channel", "conversion"])
    with pytest.raises(JourneyError, match="positive"):
        build_journeys(events, lookback_window="0D")
