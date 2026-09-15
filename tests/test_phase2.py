import pandas as pd
import pytest

from mta_audit import (
    MarkovAttribution,
    MTAAudit,
    Severity,
    build_journeys,
    calculate_channel_concentration,
    calculate_model_disagreement,
    calculate_path_sparsity,
)


@pytest.fixture
def phase2_journeys():
    events = pd.DataFrame(
        {
            "user_id": ["one", "two", "two", "three"],
            "timestamp": pd.to_datetime(
                ["2026-01-01", "2026-01-01", "2026-01-02", "2026-01-01"], utc=True
            ),
            "channel": ["A", "A", "B", "B"],
            "conversion": [1, 0, 1, 0],
        }
    )
    return build_journeys(events)


def test_markov_transition_probabilities_and_removal_are_hand_verifiable(
    phase2_journeys,
) -> None:
    model = MarkovAttribution().fit(phase2_journeys)

    probabilities = model.transition_probabilities
    assert probabilities is not None
    assert probabilities.loc["START", "A"] == pytest.approx(2 / 3)
    assert probabilities.loc["START", "B"] == pytest.approx(1 / 3)
    assert probabilities.loc["A", "CONVERSION"] == pytest.approx(1 / 2)
    assert probabilities.loc["A", "B"] == pytest.approx(1 / 2)
    assert probabilities.loc["B", "CONVERSION"] == pytest.approx(1 / 2)
    assert probabilities.loc["B", "NULL"] == pytest.approx(1 / 2)
    assert model.baseline_conversion_probability == pytest.approx(2 / 3)
    assert model.removed_conversion_probabilities_["A"] == pytest.approx(1 / 2)
    assert model.removal_effects["A"] == pytest.approx(1 / 4)
    assert model.removal_effects["B"] == 0

    result = model.attribute().data.set_index("channel")
    assert result.loc["A", "share"] == 1
    assert result["attribution_credit"].sum() == pytest.approx(2)


def test_markov_max_channels_collapses_the_long_tail_explicitly(phase2_journeys) -> None:
    model = MarkovAttribution(max_channels=1).fit(phase2_journeys)
    assert model.aggregated_channels_
    assert set(model.removal_effects_) <= {"A", "B", "__OTHER__"}
    assert model.attribute().data["share"].sum() == pytest.approx(1)


def test_markov_can_redirect_removed_transitions_to_null(phase2_journeys) -> None:
    model = MarkovAttribution(
        removal_strategy="redirect_to_null"
    ).fit(phase2_journeys)

    assert model.baseline_conversion_probability == pytest.approx(2 / 3)
    assert model.removed_conversion_probabilities["A"] == pytest.approx(1 / 6)
    assert model.removal_effects["A"] == pytest.approx(3 / 4)
    assert model.removed_conversion_probabilities["B"] == pytest.approx(1 / 3)
    assert model.removal_effects["B"] == pytest.approx(1 / 2)


def test_markov_rejects_unknown_removal_strategy() -> None:
    with pytest.raises(ValueError, match="removal_strategy"):
        MarkovAttribution(removal_strategy="unknown")


def test_phase2_metrics_and_direct_orchestrator_api(phase2_journeys) -> None:
    first = MTAAudit(
        data=pd.DataFrame(
            {
                "person": ["u", "u"],
                "at": ["2026-01-01", "2026-01-02"],
                "source": ["A", "B"],
                "goal": [0, 1],
            }
        ),
        user_col="person",
        timestamp_col="at",
        channel_col="source",
        conversion_col="goal",
    ).run()
    assert first.journey_summary.maximum_touches == 2
    assert {result.name for result in first.results} >= {
        "model_disagreement",
        "path_sparsity",
        "channel_concentration",
    }

    attribution = {
        "same_one": pd.DataFrame({"channel": ["A", "B"], "share": [0.75, 0.25]}),
        "same_two": pd.DataFrame({"channel": ["A", "B"], "share": [0.75, 0.25]}),
    }
    assert calculate_model_disagreement(attribution).agreement_score == pytest.approx(100)
    assert calculate_path_sparsity(phase2_journeys).unique_path_count == 3
    concentration = calculate_channel_concentration(phase2_journeys, attribution)
    assert concentration.touch.top1_share == pytest.approx(0.5)
    assert Severity.parse("pass") is Severity.PASS
