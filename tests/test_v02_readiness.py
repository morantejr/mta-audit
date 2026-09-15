import pandas as pd

from mta_audit import (
    MTAAudit,
    transform_synthetic_attribution,
)
from mta_audit.checks import run_data_quality_checks, run_duplicate_checks


def test_synthetic_adapter_keeps_null_paths() -> None:
    touchpoints = pd.DataFrame(
        {
            "journey_id": ["a", "a", "b"],
            "step": [1, 2, 1],
            "channel": ["paid_search", "email", "display"],
            "t_hours": [0, 24, 0],
        }
    )
    journeys = pd.DataFrame({"journey_id": ["a", "b"], "converted": [1, 0]})
    events = transform_synthetic_attribution(touchpoints, journeys)
    report = MTAAudit(data=events).run(
        attribution_models=["linear", "markov"],
        checks=["data_quality", "markov_identification"],
    )
    assert report.markov_identified is True
    assert report.results[1].status.value == "insufficient_evidence"
    assert events["conversion"].sum() == 1
    assert (~events["conversion"]).sum() == 2


def test_converter_only_markov_is_not_identified() -> None:
    events = pd.DataFrame(
        {
            "user_id": ["u", "u"],
            "timestamp": pd.to_datetime(
                ["2026-01-01 12:00:00", "2026-01-02 12:00:00"], utc=True
            ),
            "channel": ["search", "email"],
            "conversion": [False, True],
            "conversion_value": [0, 50],
        }
    )
    report = MTAAudit(data=events).run(
        attribution_models=["markov"],
        checks=["markov_identification"],
    )
    assert report.markov_identified is False
    assert report.results[0].score.value == 0


def test_zero_conversion_value_is_critical() -> None:
    events = pd.DataFrame(
        {
            "user_id": ["u", "u"],
            "timestamp": pd.to_datetime(
                ["2026-01-01 12:00:00", "2026-01-02 12:00:00"], utc=True
            ),
            "channel": ["search", "email"],
            "conversion": [False, True],
            "conversion_value": [0, 0],
        }
    )
    quality = run_data_quality_checks(events)
    assert "zero_conversion_value" in {finding.code for finding in quality.findings}


def test_date_grain_repeats_are_not_duplicate_failures() -> None:
    events = pd.DataFrame(
        {
            "user_id": ["u", "u"],
            "timestamp": ["2026-01-01", "2026-01-01"],
            "channel": ["email", "email"],
            "conversion": [False, True],
        }
    )
    duplicates = run_duplicate_checks(events)
    assert [finding.code for finding in duplicates.findings] == [
        "date_grain_repeated_touches"
    ]
    assert duplicates.findings[0].severity.name == "INFO"


def test_default_run_scores_identity_robustness() -> None:
    events = pd.DataFrame(
        {
            "user_id": ["a", "a", "b", "b"],
            "timestamp": pd.to_datetime(
                [
                    "2026-01-01 10:00:00",
                    "2026-01-02 10:00:00",
                    "2026-01-01 11:00:00",
                    "2026-01-03 11:00:00",
                ],
                utc=True,
            ),
            "channel": ["search", "email", "social", "email"],
            "conversion": [False, True, False, False],
            "conversion_value": [0, 20, 0, 0],
        }
    )
    report = MTAAudit(data=events).run(attribution_models=["linear", "markov"])
    assert "identity_robustness" in report.score.components
    assert set(report.score.components) >= {
        "data_quality",
        "window_stability",
        "identity_robustness",
        "path_quality",
        "model_agreement",
    }
    temporal = next(result for result in report.results if result.name == "model_stability")
    assert temporal.status.value == "insufficient_evidence"
    assert "temporal_stability" not in report.score.components
    assert report.markov_identified is True
    assert report.channel_grain == "channel"
    assert "Markov removal effects: identified" in report.summary()


def test_fine_channel_grain_is_flagged() -> None:
    rows = []
    for index in range(51):
        rows.append(
            {
                "user_id": f"u{index}",
                "timestamp": pd.Timestamp("2026-01-01", tz="UTC"),
                "channel": f"campaign_{index}",
                "conversion": True,
                "conversion_value": 1.0,
            }
        )
    events = pd.DataFrame(rows)
    report = MTAAudit(data=events).run(
        attribution_models=["linear"],
        checks=["path_sparsity"],
    )
    assert report.channel_grain == "campaign"
    assert "fine_channel_grain" in {
        finding.code for finding in report.results[0].findings
    }
