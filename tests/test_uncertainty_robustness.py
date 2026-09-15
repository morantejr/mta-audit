import json
from types import SimpleNamespace

import pandas as pd
import pytest

from mta_audit import (
    AttributionResult,
    AuditFinding,
    AuditReport,
    AuditResult,
    AuditScore,
    BootstrapConfig,
    BootstrapResult,
    EvidenceStatus,
    MTAAudit,
    SensitivityResult,
    calculate_reliability_score,
    make_stress_scenario,
    sample_for_audit,
    score_label,
)
from mta_audit.robustness import collect_sensitivity, evaluate_decision_robustness


@pytest.fixture
def stable_events() -> pd.DataFrame:
    return make_stress_scenario("stable_dominant").events


def test_bootstrap_is_deterministic_and_typed(stable_events: pd.DataFrame) -> None:
    first = MTAAudit(stable_events).run(
        attribution_models=["linear"],
        bootstrap=True,
        n_bootstrap=30,
        random_state=7,
    )
    second = MTAAudit(stable_events).run(
        attribution_models=["linear"],
        bootstrap=BootstrapConfig(n_iterations=30, random_state=7),
    )

    assert first.bootstrap is not None
    assert second.bootstrap is not None
    pd.testing.assert_frame_equal(first.bootstrap.draws, second.bootstrap.draws)
    stats = first.bootstrap.to_dataframe()
    assert {
        "median_share",
        "interval_low",
        "interval_high",
        "median_rank",
        "probability_top_1",
        "probability_top_2",
        "probability_top_3",
    }.issubset(stats)
    assert stats["probability_top_1"].between(0, 1).all()


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"n_iterations": 9}, "at least 10"),
        ({"confidence_level": 1.0}, "between 0 and 1"),
        ({"min_converted_journeys": 0}, "at least 1"),
    ],
)
def test_bootstrap_config_validation(kwargs: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        BootstrapConfig(**kwargs)


def test_bootstrap_abstains_on_too_few_conversions() -> None:
    events = pd.DataFrame(
        {
            "user_id": ["a", "b", "c"],
            "timestamp": pd.to_datetime(["2026-01-01"] * 3, utc=True),
            "channel": ["search", "display", "email"],
            "conversion": [True, False, False],
            "conversion_value": [1.0, 0.0, 0.0],
        }
    )
    report = MTAAudit(events).run(
        attribution_models=["linear"],
        checks=["data_quality"],
        bootstrap=True,
        n_bootstrap=10,
    )
    assert report.bootstrap is not None
    assert report.bootstrap.status == "insufficient_evidence"
    assert report.bootstrap.statistics.empty
    assert "requires at least" in (report.bootstrap.explanation or "")


def test_sensitivity_and_decision_robustness(stable_events: pd.DataFrame) -> None:
    report = MTAAudit(stable_events).run(
        attribution_models=["first_touch", "last_touch", "linear"],
        conversion_windows=[7, 30],
        bootstrap=True,
        n_bootstrap=20,
    )
    ranks = report.sensitivity_matrix("rank")
    shares = report.sensitivity_matrix("share")
    assert "baseline" in ranks
    assert set(ranks["channel"]) == set(shares["channel"])
    decision = report.decision_robustness()
    assert decision.status == "evaluated"
    leader = decision.statistics.iloc[0]
    assert leader["channel"] == "search"
    assert leader["classification"] == "Robust"
    assert "not causal truth" in decision.interpretation
    with pytest.raises(ValueError, match=r"rank.*share"):
        report.sensitivity_matrix("credit")


def test_reports_include_caveat_and_optional_sections(
    stable_events: pd.DataFrame,
) -> None:
    report = MTAAudit(stable_events).run(bootstrap=True, n_bootstrap=10)
    markdown = report.to_markdown()
    html = report.to_html()
    assert "Overall audit score" in markdown
    assert "Bootstrap uncertainty" in markdown
    assert "Decision robustness" in markdown
    assert "do not estimate causal incrementality" in markdown
    assert "<!doctype html>" in html
    assert "do not estimate causal incrementality" in html


def test_score_labels_are_qualitative_and_centralized() -> None:
    assert score_label(90) == "Robust"
    assert score_label(80) == "Stable"
    assert score_label(70) == "Caution"
    assert score_label(60) == "Fragile"
    assert score_label(59.9) == "Highly Fragile"


def test_insufficient_checks_do_not_become_zero_or_perfect_scores() -> None:
    score = calculate_reliability_score(
        [
            AuditResult(
                "model_disagreement",
                (),
                AuditScore(0),
                EvidenceStatus.INSUFFICIENT_EVIDENCE,
                "one model",
            )
        ]
    )
    assert score.status is EvidenceStatus.INSUFFICIENT_EVIDENCE
    assert score.label == "Insufficient Evidence"


def test_abstaining_check_score_is_not_labeled_highly_fragile(
    stable_events: pd.DataFrame,
) -> None:
    report = MTAAudit(stable_events).run(
        checks=["model_disagreement"],
        attribution_models=["linear"],
    )
    result = report.results[0]
    assert result.status is EvidenceStatus.INSUFFICIENT_EVIDENCE
    assert result.score.label == "Insufficient Evidence"


def test_empty_sensitivity_scenarios_are_discarded() -> None:
    baseline_frame = pd.DataFrame(
        {
            "channel": ["a", "b"],
            "attribution_credit": [2.0, 1.0],
            "share": [2 / 3, 1 / 3],
            "conversions": [2.0, 1.0],
        }
    )
    empty_frame = baseline_frame.iloc[0:0]
    result = AuditResult(
        "conversion_window",
        (
            AuditFinding(
                check="conversion_window",
                details={
                    "metrics": SimpleNamespace(
                        attribution={
                            "empty": empty_frame,
                            "valid": baseline_frame.assign(
                                share=[0.4, 0.6],
                                attribution_credit=[1.0, 2.0],
                            ),
                        }
                    )
                },
            ),
        ),
        AuditScore(80),
    )
    sensitivity = collect_sensitivity(
        {"linear": AttributionResult("linear", baseline_frame)},
        (result,),
    )
    assert "conversion_window:empty" not in sensitivity.scenarios
    assert sensitivity.status is EvidenceStatus.EVALUATED


def test_bootstrap_draws_do_not_swamp_assumption_classification() -> None:
    baseline = pd.DataFrame({"channel": ["a", "b", "c"], "share": [0.6, 0.3, 0.1]})
    reversal = pd.DataFrame({"channel": ["a", "b", "c"], "share": [0.1, 0.6, 0.3]})
    sensitivity = SensitivityResult(
        EvidenceStatus.EVALUATED,
        None,
        "baseline",
        {"baseline": baseline, "reversal": reversal},
    )
    draws = pd.DataFrame(
        [
            {
                "iteration": index,
                "model": "linear",
                "channel": channel,
                "share": share,
                "rank": rank,
            }
            for index in range(100)
            for channel, share, rank in [
                ("a", 0.6, 1.0),
                ("b", 0.3, 2.0),
                ("c", 0.1, 3.0),
            ]
        ]
    )
    bootstrap = BootstrapResult(
        EvidenceStatus.EVALUATED,
        None,
        BootstrapConfig(n_iterations=100),
        pd.DataFrame(),
        draws,
    )
    decision = evaluate_decision_robustness(sensitivity, bootstrap)
    channel_a = decision.statistics.set_index("channel").loc["a"]
    assert channel_a["probability_top_1"] == pytest.approx(0.5)
    assert channel_a["bootstrap_probability_top_1"] == pytest.approx(1.0)
    assert channel_a["classification"] != "Robust"


def test_two_channel_rank_swaps_are_not_automatically_robust() -> None:
    sensitivity = SensitivityResult(
        EvidenceStatus.EVALUATED,
        None,
        "baseline",
        {
            "baseline": pd.DataFrame(
                {"channel": ["a", "b"], "share": [0.7, 0.3]}
            ),
            "alternate": pd.DataFrame(
                {"channel": ["a", "b"], "share": [0.3, 0.7]}
            ),
        },
    )
    decision = evaluate_decision_robustness(sensitivity)
    assert not (decision.statistics["classification"] == "Robust").any()


def test_json_omits_bootstrap_draws_unless_requested(
    stable_events: pd.DataFrame,
) -> None:
    report = MTAAudit(stable_events).run(bootstrap=True, n_bootstrap=10)
    compact = json.loads(report.to_json())
    expanded = json.loads(report.to_json(include_bootstrap_draws=True))
    assert "draws" not in compact["bootstrap"]
    assert len(expanded["bootstrap"]["draws"]) > 0


def test_unevaluated_decision_result_has_stable_schema() -> None:
    report = AuditReport(results=(), score=AuditScore(0))
    frame = report.decision_robustness().to_dataframe()
    assert {"channel", "classification", "probability_top_2"}.issubset(frame.columns)


def test_sample_for_audit_preserves_complete_users_and_metadata(
    stable_events: pd.DataFrame,
) -> None:
    sampled = sample_for_audit(
        stable_events,
        max_journeys=20,
        random_state=5,
    )
    selected = set(sampled["user_id"])
    expected = stable_events.loc[stable_events["user_id"].isin(selected)]
    pd.testing.assert_frame_equal(
        sampled.reset_index(drop=True),
        expected.reset_index(drop=True),
    )
    report = MTAAudit(sampled).run(checks=["data_quality"])
    assert report.sample_info is not None
    assert report.sample_info["sampled"] is True
    assert report.sample_info["output_users"] == 20


@pytest.mark.parametrize(
    "name",
    [
        "stable_dominant",
        "window_sensitive",
        "identity_fragmentation",
        "duplicate_contamination",
        "model_disagreement",
        "noise_without_decision_change",
    ],
)
def test_stress_scenarios_are_deterministic_and_documented(name: str) -> None:
    first = make_stress_scenario(name, random_state=11)
    second = make_stress_scenario(name, random_state=11)
    pd.testing.assert_frame_equal(first.events, second.events)
    assert first.purpose
    assert first.expected_behavior


def test_known_conditions_trigger_expected_diagnostics() -> None:
    stable = MTAAudit(make_stress_scenario("stable_dominant").events).run(
        attribution_models=["first_touch", "last_touch", "linear"],
        bootstrap=True,
        n_bootstrap=10,
    )
    stable_leader = stable.decision_robustness().statistics.iloc[0]
    assert stable_leader["channel"] == "search"
    assert stable_leader["classification"] == "Robust"

    window = MTAAudit(make_stress_scenario("window_sensitive").events).run(
        checks=["conversion_window"],
        conversion_windows=[7, 30],
    )
    assert window.results[0].score.value < 80

    duplicate = MTAAudit(
        make_stress_scenario("duplicate_contamination").events
    ).run(checks=["duplicates"])
    assert duplicate.results[0].score.value < 100
    assert duplicate.results[0].findings
    assert "duplicate_handling:deduplicated" in duplicate.sensitivity_matrix("share")

    identity_data = make_stress_scenario("identity_fragmentation").events
    identity = MTAAudit(identity_data).simulate_identity_loss(rates=[0.2])
    assert identity.score.value < 100

    disagreement = MTAAudit(
        make_stress_scenario("model_disagreement").events
    ).run(
        checks=["model_disagreement"],
        attribution_models=["first_touch", "last_touch", "linear", "markov"],
    )
    assert disagreement.results[0].score.value < 60

    noise = MTAAudit(
        make_stress_scenario("noise_without_decision_change").events
    ).run(
        attribution_models=["linear"],
        checks=["conversion_window", "identity_loss", "touchpoint_loss"],
        bootstrap=True,
        n_bootstrap=10,
    )
    noise_leader = noise.decision_robustness().statistics.iloc[0]
    assert noise_leader["channel"] == "search"
    assert noise_leader["classification"] == "Robust"


def test_full_observational_stability_flow(stable_events: pd.DataFrame) -> None:
    report = MTAAudit(stable_events).run(
        attribution_models=["first_touch", "last_touch", "linear", "markov"],
        conversion_windows=[7, 14, 30],
        simulations={"touchpoint_loss": {"rates": [0.1]}},
        bootstrap=BootstrapConfig(n_iterations=20, random_state=42),
    )
    assert report.journey_summary is not None
    assert report.attribution
    assert report.bootstrap is not None and report.bootstrap.evaluated
    assert report.bootstrap.config.stratify_conversion is True
    assert not report.sensitivity_matrix().empty
    assert report.decision_robustness().status == "evaluated"
    assert report.score.components
    assert report.to_markdown().startswith("# MTA Audit")
