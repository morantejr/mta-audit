import runpy
from pathlib import Path
from typing import Any, cast

NAMESPACE = runpy.run_path(
    str(Path(__file__).parents[1] / "benchmarks" / "scoring_sensitivity.py")
)
BANDS = cast(dict[str, tuple[float, float, float, float]], NAMESPACE["BANDS"])
WEIGHT_SCENARIOS = cast(dict[str, dict[str, float]], NAMESPACE["WEIGHT_SCENARIOS"])
label = cast(Any, NAMESPACE["label"])
weighted_score = cast(Any, NAMESPACE["weighted_score"])


def test_weight_scenarios_normalize_only_available_components() -> None:
    components = {
        "data_quality": 59.25,
        "path_quality": 61.81,
        "model_agreement": 93.34,
    }
    scores = {
        name: weighted_score(components, weights)
        for name, weights in WEIGHT_SCENARIOS.items()
    }
    assert scores["data-quality-heavy"] < scores["default"]
    assert scores["model-agreement-heavy"] > scores["default"]
    assert min(components.values()) <= scores["default"] <= max(components.values())


def test_rating_policy_can_change_label_without_changing_score() -> None:
    score = 72.35
    assert label(score, BANDS["default"]) == "Caution"
    assert label(score, BANDS["five-points-stricter"]) == "Fragile"
    assert label(score, BANDS["five-points-looser"]) == "Caution"
