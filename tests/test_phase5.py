import builtins
import json

import matplotlib
import pandas as pd
import pytest

from mta_audit import (
    AuditResult,
    AuditScore,
    MTAAudit,
    ScoringWeights,
    calculate_reliability_score,
)

matplotlib.use("Agg")


@pytest.fixture
def events() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "user_id": ["a", "a", "a", "b", "b", "c", "c"],
            "timestamp": pd.to_datetime(
                [
                    "2026-01-01",
                    "2026-01-02",
                    "2026-01-03",
                    "2026-01-01",
                    "2026-01-08",
                    "2026-01-02",
                    "2026-01-15",
                ]
            ),
            "channel": ["search", "email", "direct", "social", "direct", "email", "search"],
            "conversion": [0, 0, 1, 0, 1, 0, 1],
        }
    )


def test_scoring_normalizes_only_available_components() -> None:
    quality = AuditResult("data_quality", (), AuditScore(50))
    agreement = AuditResult("model_disagreement", (), AuditScore(100))
    score = calculate_reliability_score(
        [quality, agreement],
        ScoringWeights(data_quality=3, model_agreement=1),
    )

    assert score.value == pytest.approx(62.5)
    assert score.components == {"data_quality": 50, "model_agreement": 100}
    assert score.weights == {"data_quality": 0.75, "model_agreement": 0.25}
    assert "window_stability" not in score.components


def test_report_serialization_and_summary(events: pd.DataFrame) -> None:
    report = MTAAudit(data=events).run(checks=["data_quality", "model_disagreement"])

    assert "Overall audit score:" in report.summary()
    assert "Data Quality:" in report.summary()
    assert "Model Agreement:" in report.summary()
    frame = report.to_dataframe()
    assert {"check", "severity", "check_score", "report_score"}.issubset(frame)
    payload = json.loads(report.to_json())
    assert payload["score"]["components"]
    assert isinstance(payload["attribution"]["linear"], dict)


def test_advanced_run_api_and_plots(events: pd.DataFrame) -> None:
    report = MTAAudit(data=events, scoring_weights={"window_stability": 2}).run(
        checks=["data_quality", "conversion_window", "model_disagreement"],
        attribution_models=["first_touch", "linear", "markov"],
        conversion_windows=[1, 7, 30],
        simulations={"identity_loss": {"rates": [0.1], "models": ["linear"]}},
    )

    assert set(report.attribution) == {"first_touch", "linear", "markov"}
    assert "identity_robustness" in report.score.components
    assert report.plot_scorecard().figure is not None
    assert report.plot_model_comparison().figure is not None
    assert report.plot_window_sensitivity().figure is not None
    assert report.plot_channel_volatility().figure is not None


def test_plotting_dependency_error_is_helpful(
    events: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
) -> None:
    report = MTAAudit(data=events).run(checks=["data_quality"])
    original_import = builtins.__import__

    def reject_matplotlib(name: str, *args: object, **kwargs: object) -> object:
        if name == "matplotlib.pyplot":
            raise ImportError("simulated missing extra")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", reject_matplotlib)
    with pytest.raises(ImportError, match=r"mta-audit\[viz\]"):
        report.plot_scorecard()


def test_configuration_validation_and_missing_plot_data(events: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="Unknown scoring weights"):
        MTAAudit(data=events, scoring_weights={"mystery": 1})
    with pytest.raises(ValueError, match="positive"):
        MTAAudit(data=events, scoring_weights={key: 0 for key in ScoringWeights().as_dict()})

    report = MTAAudit(data=events).run(checks=["data_quality"])
    with pytest.raises(ValueError, match="conversion_window"):
        report.plot_window_sensitivity()
