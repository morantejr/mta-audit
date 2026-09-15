"""Deterministic known-condition scenarios for audit-framework validation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True, slots=True)
class SyntheticStressScenario:
    """Controlled observational paths with expected diagnostic behavior."""

    name: str
    purpose: str
    expected_behavior: tuple[str, ...]
    events: pd.DataFrame
    random_state: int


def make_stress_scenario(
    name: str,
    *,
    random_state: int = 42,
) -> SyntheticStressScenario:
    """Return one of six deterministic observational stress scenarios."""

    normalized = name.strip().lower().replace("-", "_").replace(" ", "_")
    builders = {
        "stable_dominant": _stable_dominant,
        "window_sensitive": _window_sensitive,
        "identity_fragmentation": _identity_fragmentation,
        "duplicate_contamination": _duplicate_contamination,
        "model_disagreement": _model_disagreement,
        "noise_without_decision_change": _noise_without_decision_change,
    }
    try:
        return builders[normalized](random_state)
    except KeyError as exc:
        raise ValueError(
            f"Unknown stress scenario {name!r}. Available: {sorted(builders)}"
        ) from exc


def _events(paths: list[tuple[list[str], bool]], *, gap_days: int = 1) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    origin = pd.Timestamp("2026-01-01", tz="UTC")
    for user, (channels, converted) in enumerate(paths):
        endpoint = origin + pd.Timedelta(days=user % 35 + len(channels) * gap_days)
        for step, channel in enumerate(channels):
            timestamp = endpoint - pd.Timedelta(days=(len(channels) - step - 1) * gap_days)
            is_conversion = converted and step == len(channels) - 1
            rows.append(
                {
                    "user_id": f"u{user}",
                    "timestamp": timestamp,
                    "channel": channel,
                    "conversion": is_conversion,
                    "conversion_value": 1.0 if is_conversion else 0.0,
                }
            )
    return pd.DataFrame(rows)


def _stable_dominant(seed: int) -> SyntheticStressScenario:
    paths = [(["search", "search"], True) for _ in range(80)]
    paths += [(["display", "search"], True) for _ in range(10)]
    paths += [(["social"], False) for _ in range(20)]
    return SyntheticStressScenario(
        "stable_dominant",
        "A dominant channel should survive windows, models, and sampling.",
        ("high window stability", "strong model agreement", "robust top ranking"),
        _events(paths),
        seed,
    )


def _window_sensitive(seed: int) -> SyntheticStressScenario:
    paths = [(["social", "search"], True) for _ in range(70)]
    paths += [(["email"], True) for _ in range(20)]
    paths += [(["display"], False) for _ in range(20)]
    return SyntheticStressScenario(
        "window_sensitive",
        "Long-lag upper-funnel touches disappear under a short window.",
        ("lower window stability", "rank changes across windows"),
        _events(paths, gap_days=20),
        seed,
    )


def _identity_fragmentation(seed: int) -> SyntheticStressScenario:
    paths = [(["social", "display", "search"], True) for _ in range(60)]
    paths += [(["email", "search"], True) for _ in range(20)]
    paths += [(["social", "display"], False) for _ in range(20)]
    return SyntheticStressScenario(
        "identity_fragmentation",
        "Upper-funnel credit should move when multi-touch identities fragment.",
        ("journey fragmentation", "ranking drift", "lower identity robustness"),
        _events(paths),
        seed,
    )


def _duplicate_contamination(seed: int) -> SyntheticStressScenario:
    base = _events(
        [(["display", "search"], True) for _ in range(30)]
        + [(["email", "search"], True) for _ in range(30)]
    )
    display = base.loc[
        (base["channel"] == "display") & ~base["conversion"]
    ].copy()
    contaminated = pd.concat([base, display, display], ignore_index=True)
    return SyntheticStressScenario(
        "duplicate_contamination",
        "Exact duplicate touches from one channel should be detected.",
        ("duplicate finding", "lower data quality", "deduplication share shift"),
        contaminated,
        seed,
    )


def _model_disagreement(seed: int) -> SyntheticStressScenario:
    paths = [(["social", "display", "search"], True) for _ in range(60)]
    paths += [(["email", "search"], True) for _ in range(20)]
    paths += [(["display", "email"], False) for _ in range(30)]
    return SyntheticStressScenario(
        "model_disagreement",
        "First, last, linear, and removal-effect models favor different positions.",
        ("lower model agreement", "volatile ranks"),
        _events(paths),
        seed,
    )


def _noise_without_decision_change(seed: int) -> SyntheticStressScenario:
    rng = np.random.default_rng(seed)
    minor = np.asarray(["display", "social", "email"])
    paths: list[tuple[list[str], bool]] = []
    for _ in range(100):
        paths.append(([str(rng.choice(minor)), "search", "search"], True))
    paths += [([str(rng.choice(minor))], False) for _ in range(30)]
    return SyntheticStressScenario(
        "noise_without_decision_change",
        "Minor-channel noise should not overturn a clearly dominant channel.",
        ("some metric movement", "robust dominant decision"),
        _events(paths),
        seed,
    )
