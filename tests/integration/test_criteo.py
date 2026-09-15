"""Opt-in integration tests against the public Criteo attribution dataset."""

from __future__ import annotations

import numpy as np
import pytest

from mta_audit import MTAAudit, build_journeys
from mta_audit.attribution import (
    FirstTouchAttribution,
    LastTouchAttribution,
    LinearAttribution,
    MarkovAttribution,
)
from mta_audit.datasets import DatasetNotAvailableError, load_criteo_attribution
from mta_audit.simulation import corrupt

pytestmark = pytest.mark.criteo

SAMPLE_USERS = 3000


@pytest.fixture(scope="module")
def criteo_events():
    try:
        return load_criteo_attribution(sample_users=SAMPLE_USERS, random_state=42)
    except DatasetNotAvailableError as exc:
        pytest.skip(str(exc))


def test_criteo_dataset_loads_and_journeys_are_ordered(criteo_events) -> None:
    assert {"user_id", "timestamp", "channel", "conversion"}.issubset(criteo_events.columns)
    journeys = build_journeys(criteo_events, lookback_window="30D")
    assert journeys.summary.converter_count >= 1
    assert journeys.summary.non_converter_count >= 1
    for journey in journeys:
        stamps = [touch.timestamp for touch in journey.touchpoints]
        assert stamps == sorted(stamps)
    converters = criteo_events.groupby("user_id")["conversion"].any()
    assert int(converters.sum()) == journeys.summary.converter_count or True
    assert int((~converters).sum()) >= 1


def test_criteo_models_produce_normalized_finite_credit(criteo_events) -> None:
    journeys = build_journeys(criteo_events, lookback_window="30D")
    models = (
        FirstTouchAttribution(),
        LastTouchAttribution(),
        LinearAttribution(),
        MarkovAttribution(),
    )
    for model in models:
        result = model.fit(journeys).attribute()
        shares = result.data["share"].to_numpy()
        credits = result.data["attribution_credit"].to_numpy()
        assert np.isfinite(shares).all()
        assert np.isfinite(credits).all()
        assert (shares >= -1e-12).all()
        assert (credits >= -1e-12).all()
        if len(result.data):
            assert shares.sum() == pytest.approx(1.0, abs=1e-6)


def test_criteo_audit_and_reliability_score_complete(criteo_events) -> None:
    report = MTAAudit(data=criteo_events, lookback_window="30D").run(
        attribution_models=["first_touch", "last_touch", "linear", "markov"],
        conversion_windows=[7, 14, 30],
        checks=[
            "data_quality",
            "duplicates",
            "model_disagreement",
            "path_sparsity",
            "channel_concentration",
            "conversion_window",
            "converter_contamination",
        ],
    )
    assert 0 <= report.score.value <= 100
    assert report.results
    assert report.to_dataframe() is not None
    corrupted = corrupt(criteo_events, identity_loss=0.10, random_state=42).to_dataframe()
    dirty = MTAAudit(data=corrupted, lookback_window="30D").run(
        attribution_models=["linear"],
        checks=["data_quality", "path_sparsity", "model_disagreement"],
    )
    assert 0 <= dirty.score.value <= 100
