import json

import pandas as pd

from mta_audit import (
    AuditFinding,
    AuditReport,
    AuditResult,
    AuditScore,
    ColumnMapping,
    MTAAudit,
    build_journeys,
    run_data_quality_checks,
    run_duplicate_checks,
    validate_events,
)


def test_flexible_schema_preserves_extra_columns_and_campaign_alias() -> None:
    source = pd.DataFrame(
        {
            "person": ["u", "u"],
            "at": ["2026-01-01", "2026-01-02"],
            "medium": ["email", "direct"],
            "sale": [0, 1],
            "converted_at": [None, "2026-01-02"],
            "campaign_key": ["c1", "c1"],
            "is_impression": [True, False],
            "is_click": [False, True],
            "order": [None, "o1"],
            "diagnostic_payload": ["keep", "me"],
        }
    )
    mapping = ColumnMapping(
        user_id="person",
        timestamp="at",
        channel="medium",
        conversion="sale",
        conversion_timestamp="converted_at",
        campaign_id="campaign_key",
        impression="is_impression",
        click="is_click",
        order_id="order",
    )

    normalized, _ = validate_events(source, mapping)

    assert normalized["campaign_id"].tolist() == ["c1", "c1"]
    assert normalized["campaign"].tolist() == ["c1", "c1"]
    assert normalized["diagnostic_payload"].tolist() == ["keep", "me"]


def test_metric_value_alias_serializes_both_names() -> None:
    finding = AuditFinding(check="example", metric_name="rate", metric_value=0.25)
    assert finding.value == finding.metric_value == 0.25
    report = AuditReport(
        (AuditResult("example", (finding,), AuditScore(100)),),
        AuditScore(100),
    )

    payload = json.loads(report.to_json())
    assert payload["results"][0]["findings"][0]["metric_value"] == 0.25
    assert report.to_dataframe().iloc[0]["metric_value"] == 0.25


def test_boolean_event_loss_needs_no_event_type_column() -> None:
    events = pd.DataFrame(
        {
            "user_id": ["u", "u", "u"],
            "timestamp": pd.date_range("2026-01-01", periods=3, tz="UTC"),
            "channel": ["display", "search", "direct"],
            "conversion": [False, False, True],
            "impression": [True, False, False],
            "click": [False, True, False],
        }
    )
    audit = MTAAudit(
        events,
        impression_col="impression",
        click_col="click",
    )

    result = audit.simulate_event_loss(impression_loss=1, click_loss=1)

    metadata = result.findings[0].details["metrics"].scenarios[0].metadata
    assert metadata["row_counts"]["impression_loss_rows_removed"] == 1
    assert dict(metadata)["row_counts"]["click_loss_rows_removed"] == 1


def test_quality_and_duplicate_diagnostics_are_structured() -> None:
    events = pd.DataFrame(
        {
            "user_id": ["u", "u", "u", "u"],
            "timestamp": [
                "2026-01-02",
                "2026-01-02 00:00:00.500",
                "2026-01-03",
                "2026-01-03",
            ],
            "channel": ["email", "email", "direct", "direct"],
            "conversion": [0, 0, 1, 1],
            "conversion_timestamp": [None, None, "2026-01-01", "2026-01-01"],
            "order_id": [None, None, "order-1", "order-1"],
        }
    )
    mapping = {
        "conversion_timestamp": "conversion_timestamp",
        "order_id": "order_id",
    }

    quality = run_data_quality_checks(
        events, mapping, reference_time="2026-12-31"
    )
    duplicates = run_duplicate_checks(events, mapping)

    assert {"conversion_before_exposure", "repeated_conversion_ids"} <= {
        finding.code for finding in quality.findings
    }
    near = next(finding for finding in duplicates.findings if finding.code == "near_duplicates")
    assert near.metric_name == "duplicate_event_rate"
    assert {"duplicate_event_rate", "affected_users", "affected_conversions"} <= set(
        near.details
    )


def test_journey_summary_contains_requested_path_metrics() -> None:
    events = pd.DataFrame(
        {
            "user_id": ["buyer", "buyer", "buyer", "browser"],
            "timestamp": pd.date_range("2026-01-01", periods=4, tz="UTC"),
            "channel": ["email", "email", "direct", "social"],
            "conversion": [0, 0, 1, 0],
        }
    )

    summary = build_journeys(events).summary

    assert summary.path_count == 2
    assert summary.converter_average_path_length == 3
    assert summary.non_converter_average_path_length == 1
    assert summary.channel_frequency["email"] == 2
    assert summary.repeated_touch_rate == 0.25
