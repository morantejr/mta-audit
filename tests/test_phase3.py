import pandas as pd
import pytest

from mta_audit import (
    LinearAttribution,
    MTAAudit,
    audit_exposure_comparison,
    audit_model_stability,
    build_journeys,
)


def test_conversion_window_rebuilds_and_reports_channel_changes() -> None:
    events = pd.DataFrame(
        {
            "user_id": ["one", "one", "two", "two"],
            "timestamp": ["2026-01-01", "2026-01-21", "2026-01-20", "2026-01-21"],
            "channel": ["early", "late", "early", "early"],
            "conversion": [0, 1, 0, 1],
        }
    )

    result = MTAAudit(events).check_conversion_window(windows=[7, 30], model="linear")
    metrics = result.findings[0].details["metrics"]
    comparison = metrics.channel_comparison.set_index("channel")

    assert metrics.windows == ("7D", "30D")
    assert comparison.loc["late", "credit__7D"] == pytest.approx(1)
    assert comparison.loc["late", "credit__30D"] == pytest.approx(0.5)
    assert comparison.loc["early", "absolute_credit_change"] == pytest.approx(0.5)
    assert metrics.max_share_volatility > 0


def test_delayed_converter_contamination_classifies_from_first_exposure() -> None:
    events = pd.DataFrame(
        {
            "user_id": ["short", "short", "delayed", "delayed", "late", "late", "none"],
            "timestamp": [
                "2026-01-01", "2026-01-04",
                "2026-01-01", "2026-01-11",
                "2026-01-01", "2026-02-10",
                "2026-01-01",
            ],
            "channel": ["email", "email", "search", "search", "social", "social", "email"],
            "campaign": ["a", "a", "b", "b", "c", "c", "a"],
            "conversion": [0, 1, 0, 1, 0, 1, 0],
        }
    )

    result = MTAAudit(events, campaign_col="campaign").check_delayed_converter_contamination()
    metrics = result.findings[0].details["metrics"]

    assert metrics.short_window_converters == 1
    assert metrics.delayed_converters == 1
    assert metrics.post_reference_converters == 1
    assert metrics.observed_non_converters == 1
    assert metrics.delayed_rate_all_users == pytest.approx(0.25)
    assert metrics.campaign_breakdown is not None


def test_temporal_stability_and_exposure_basics() -> None:
    events = pd.DataFrame(
        {
            "user_id": ["w1c", "w1c", "w1n", "w2c", "w2c", "w2n"],
            "timestamp": [
                "2026-01-01", "2026-01-02", "2026-01-03",
                "2026-01-08", "2026-01-09", "2026-01-10",
            ],
            "channel": ["email", "email", "search", "social", "social", "email"],
            "campaign": ["a", "a", "b", "c", "c", "a"],
            "conversion": [0, 1, 0, 0, 1, 0],
        }
    )

    temporal = audit_model_stability(
        build_journeys(events), model=LinearAttribution(), period="W"
    )
    temporal_metrics = temporal.findings[0].details["metrics"]
    exposure = audit_exposure_comparison(events)
    exposure_metrics = exposure.findings[0].details["metrics"]

    assert len(temporal_metrics.period_summary) == 2
    assert temporal_metrics.max_share_volatility > 0
    assert temporal_metrics.conversion_rate_drift == pytest.approx(0)
    assert exposure_metrics.converters.user_count == 2
    assert exposure_metrics.non_converters.user_count == 2
    assert exposure.findings[0].details["causal"] is False


def test_run_accepts_phase3_named_aliases() -> None:
    events = pd.DataFrame(
        {
            "user_id": ["u", "u"],
            "timestamp": ["2026-01-01", "2026-01-02"],
            "channel": ["email", "email"],
            "conversion": [0, 1],
        }
    )
    report = MTAAudit(events).run(
        checks=["conversion_window", "converter_contamination", "temporal_stability"]
    )
    assert [result.name for result in report.results] == [
        "conversion_window",
        "converter_contamination",
        "model_stability",
    ]
