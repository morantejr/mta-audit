"""Reproduce aggregate-score sensitivity from the committed Criteo benchmark."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"

WEIGHT_SCENARIOS: dict[str, dict[str, float]] = {
    "default": {
        "data_quality": 0.20,
        "window_stability": 0.15,
        "identity_robustness": 0.15,
        "path_quality": 0.15,
        "model_agreement": 0.20,
        "temporal_stability": 0.15,
    },
    "equal": {
        "data_quality": 1,
        "window_stability": 1,
        "identity_robustness": 1,
        "path_quality": 1,
        "model_agreement": 1,
        "temporal_stability": 1,
    },
    "data-quality-heavy": {
        "data_quality": 0.50,
        "window_stability": 0.10,
        "identity_robustness": 0.10,
        "path_quality": 0.10,
        "model_agreement": 0.10,
        "temporal_stability": 0.10,
    },
    "model-agreement-heavy": {
        "data_quality": 0.10,
        "window_stability": 0.10,
        "identity_robustness": 0.10,
        "path_quality": 0.10,
        "model_agreement": 0.50,
        "temporal_stability": 0.10,
    },
}

BANDS: dict[str, tuple[float, float, float, float]] = {
    "default": (90, 80, 70, 60),
    "five-points-stricter": (95, 85, 75, 65),
    "five-points-looser": (85, 75, 65, 55),
}


def weighted_score(components: dict[str, float], weights: dict[str, float]) -> float:
    """Score only available components, matching library normalization."""

    available = {name: weight for name, weight in weights.items() if name in components}
    denominator = sum(available.values())
    return sum(components[name] * weight / denominator for name, weight in available.items())


def label(value: float, bands: tuple[float, float, float, float]) -> str:
    """Apply an explicit interpretation-band scenario."""

    robust, stable, caution, fragile = bands
    if value >= robust:
        return "Robust"
    if value >= stable:
        return "Stable"
    if value >= caution:
        return "Caution"
    if value >= fragile:
        return "Fragile"
    return "Highly Fragile"


def render() -> str:
    payload = json.loads((RESULTS / "criteo_benchmark.json").read_text())
    components: dict[str, float] = payload["baseline_components"]
    scores = {
        name: weighted_score(components, weights)
        for name, weights in WEIGHT_SCENARIOS.items()
    }
    default_score = scores["default"]
    lines = [
        "# Score-policy sensitivity",
        "",
        "This is a deterministic policy sensitivity check over the committed Criteo",
        "baseline components. It does not recalibrate component checks or claim that",
        "one policy is objectively correct.",
        "",
        "Available Criteo components in this benchmark run:",
    ]
    lines.extend(f"- `{name}`: **{value:.2f}**" for name, value in components.items())
    lines.extend(["", "## Weight sensitivity", ""])
    lines.extend(f"- **{name}: {score:.2f}**" for name, score in scores.items())
    spread = max(scores.values()) - min(scores.values())
    lines.extend(
        [
            "",
            f"Policy-weight spread: **{spread:.2f} points**. The default result",
            f"({default_score:.2f}) is not invariant to stakeholder priorities;",
            "the component scores must therefore remain visible beside it.",
            "",
            "## Interpretation-band sensitivity",
            "",
        ]
    )
    lines.extend(
        f"- **{name}: {label(default_score, bands)}** ({bands})"
        for name, bands in BANDS.items()
    )
    lines.extend(
        [
            "",
            "The qualitative label can change while the measured component evidence",
            "does not. Labels are communication policy, not statistical confidence.",
            "",
            "Reproduce with `python benchmarks/scoring_sensitivity.py`.",
        ]
    )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    report = render()
    (RESULTS / "scoring_sensitivity.md").write_text(report)
    print(report)
