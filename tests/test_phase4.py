import pandas as pd
import pytest

from mta_audit import MTAAudit, corrupt, deduplicate, detect_duplicates


@pytest.fixture
def events() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "user_id": ["a", "a", "a", "b", "b", "b", "c", "c"],
            "timestamp": pd.to_datetime(
                [
                    "2026-01-01",
                    "2026-01-02",
                    "2026-01-03",
                    "2026-01-01",
                    "2026-01-02",
                    "2026-01-04",
                    "2026-01-01",
                    "2026-01-02",
                ]
            ),
            "channel": [
                "search",
                "email",
                "direct",
                "social",
                "search",
                "direct",
                "email",
                "email",
            ],
            "conversion": [0, 0, 1, 0, 0, 1, 0, 0],
            "event_type": [
                "impression",
                "click",
                "conversion",
                "impression",
                "click",
                "conversion",
                "impression",
                "click",
            ],
        }
    )


def test_corrupt_is_deterministic_preserves_input_and_records_metadata(
    events: pd.DataFrame,
) -> None:
    original = events.copy(deep=True)
    first = corrupt(
        events,
        identity_loss=1,
        touchpoint_loss=0.25,
        duplicate_rate=0.25,
        conversion_delay_shift="1D",
        timestamp_noise="1s",
        random_state=7,
    )
    second = corrupt(
        events,
        identity_loss=1,
        touchpoint_loss=0.25,
        duplicate_rate=0.25,
        conversion_delay_shift="1D",
        timestamp_noise="1s",
        random_state=7,
    )

    pd.testing.assert_frame_equal(events, original)
    pd.testing.assert_frame_equal(first.data, second.data)
    assert first.metadata == second.metadata
    assert first.metadata.requested["identity_loss"] == 1
    assert first.metadata.row_counts["initial_rows"] == len(events)
    assert first.metadata.row_counts["final_rows"] == len(first)
    assert first.metadata.row_counts["conversion_timestamps_shifted"] == 2
    assert first.metadata.affected_users["identity_loss"]
    assert any(str(value).startswith("__mta_fragment__") for value in first["user_id"])


def test_generic_and_event_specific_loss_protect_conversions(events: pd.DataFrame) -> None:
    generic = corrupt(events, touchpoint_loss=1)
    typed = corrupt(
        events,
        impression_loss=1,
        click_loss=1,
        event_type_col="event_type",
    )

    assert generic["conversion"].sum() == events["conversion"].sum()
    assert typed["conversion"].sum() == events["conversion"].sum()
    assert set(typed["event_type"]) == {"conversion"}
    assert typed.metadata.row_counts["impression_loss_rows_removed"] == 3
    assert typed.metadata.row_counts["click_loss_rows_removed"] == 3


def test_missing_channels_delay_noise_and_edge_validation(events: pd.DataFrame) -> None:
    result = corrupt(
        events,
        missing_channel_rate=1,
        conversion_delay_shift="-1h",
        timestamp_noise="10ms",
    )
    assert result["channel"].isna().all()
    assert result.metadata.applied["conversion_delay_shift"] == pd.Timedelta("-1h")
    with pytest.raises(ValueError):
        corrupt(events, duplicate_rate=1.1)
    with pytest.raises(ValueError):
        corrupt(events, impression_loss=0.5)


def test_exact_near_and_repeated_duplicate_utilities(events: pd.DataFrame) -> None:
    exact = corrupt(events, duplicate_rate=1, duplicate_kind="exact")
    near = corrupt(events, duplicate_rate=1, duplicate_kind="near")

    exact_flags = detect_duplicates(exact.data, event_type_col="event_type")
    near_flags = detect_duplicates(
        near.data, event_type_col="event_type", near_tolerance="2us"
    )
    assert exact_flags["is_exact"].all()
    assert near_flags["is_near"].any()
    assert exact_flags["is_repeated"].all()
    assert len(deduplicate(exact.data)) == len(events)
    assert len(
        deduplicate(
            near.data,
            event_type_col="event_type",
            near_tolerance="2us",
        )
    ) == len(events)


def test_identity_touchpoint_and_event_loss_audits(events: pd.DataFrame) -> None:
    audit = MTAAudit(events)
    identity = audit.simulate_identity_loss(rates=[0, 1], models=["linear"])
    touchpoint = audit.simulate_touchpoint_loss(rate=1, models="linear")
    event = audit.simulate_event_loss(
        impression_loss=1,
        click_loss=1,
        event_type_col="event_type",
        models=["first_touch", "last_touch"],
    )

    identity_metrics = identity.findings[0].details["metrics"]
    touchpoint_metrics = touchpoint.findings[0].details["metrics"]
    event_metrics = event.findings[0].details["metrics"]
    assert len(identity_metrics.scenarios) == 2
    assert identity_metrics.scenarios[1].fragmentation_rate > 0
    assert identity_metrics.scenarios[1].path_length_change != 0
    assert touchpoint_metrics.scenarios[0].metadata.row_counts[
        "touchpoint_loss_rows_removed"
    ] == 6
    assert event_metrics.scenarios[0].max_channel_share_drift >= 0
    pd.testing.assert_frame_equal(events, events.copy())


def test_run_accepts_configured_simulations(events: pd.DataFrame) -> None:
    report = MTAAudit(events).run(
        checks=["data_quality"],
        simulations={
            "identity_loss": {"rates": [0.5], "models": ["linear"]},
            "touchpoint_loss": 0.5,
            "event_loss": {
                "impression_loss": 1,
                "click_loss": 0,
                "event_type_col": "event_type",
                "models": ["linear"],
            },
        },
    )
    assert [result.name for result in report.results] == [
        "data_quality",
        "identity_loss",
        "touchpoint_loss",
        "event_loss",
    ]
