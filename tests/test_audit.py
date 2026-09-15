import pandas as pd

from mta_audit import MTAAudit, Severity, run_data_quality_checks, run_duplicate_checks


def test_checks_find_quality_and_duplicate_problems() -> None:
    events = pd.DataFrame(
        {
            "user_id": ["u", "u", "u"],
            "timestamp": ["bad", "2026-01-01", "2026-01-01"],
            "channel": ["", "email", "email"],
            "conversion": [0, 1, 1],
        }
    )

    quality = run_data_quality_checks(events)
    duplicates = run_duplicate_checks(events)

    assert {finding.code for finding in quality.findings} == {
        "invalid_timestamp",
        "empty_channel",
    }
    assert [finding.code for finding in duplicates.findings] == [
        "date_grain_repeated_touches"
    ]
    assert quality.score.value < 100


def test_orchestrator_returns_structured_report_and_filters_visibility() -> None:
    events = pd.DataFrame(
        {
            "visitor": ["u", "u", "u"],
            "time": ["2026-01-01", "2026-01-02", "2026-01-02"],
            "campaign": ["search", "direct", "direct"],
            "goal": [0, 1, 1],
        }
    )
    audit = MTAAudit(
        columns={
            "user_id": "visitor",
            "timestamp": "time",
            "channel": "campaign",
            "conversion": "goal",
        },
        minimum_severity="high",
    )

    report = audit.run(
        events,
        checks=["data_quality", "duplicates"],
    )

    assert report.minimum_severity is Severity.HIGH
    assert report.findings == ()
    assert report.results[1].findings[0].code == "date_grain_repeated_touches"
    assert set(report.attribution) == {"first_touch", "last_touch", "linear"}
    assert report.journey_summary.converter_count == 2
