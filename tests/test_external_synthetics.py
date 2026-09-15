from pathlib import Path

import pandas as pd
import pytest

from mta_audit import LastTouchAttribution, MTAAudit, build_journeys
from mta_audit.datasets import (
    DatasetNotAvailableError,
    load_ignazio_attribution,
    simulate_criteo_label_events,
    simulate_jd_mta_events,
    simulate_synthesizer_events,
    transform_ignazio_touchpoints,
    transform_synthesizer_touchpoints,
    transform_triangulation_mta,
)
from mta_audit.schema import SchemaError

FIXTURES = Path(__file__).parent / "fixtures"


def test_jd_simulation_is_deterministic_with_converters_and_null_paths() -> None:
    first = simulate_jd_mta_events(80, random_state=7)
    second = simulate_jd_mta_events(80, random_state=7)
    pd.testing.assert_frame_equal(first, second)
    assert first["conversion"].any()
    converters = first.loc[first["conversion"], "user_id"]
    assert (~first["user_id"].isin(converters)).any()
    report = MTAAudit(data=first).run(
        attribution_models=["linear", "markov"],
        checks=["data_quality", "markov_identification"],
    )
    assert report.markov_identified is True


def test_criteo_label_simulation_last_touch_can_credit_non_trigger_channel() -> None:
    events = simulate_criteo_label_events(500, random_state=3)
    converting = events[events["conversion"]]
    assert converting["channel"].eq("type_B").any()
    assert converting["true_trigger_channel"].eq("type_A").any()
    last = LastTouchAttribution().attribute(build_journeys(events)).data
    shares = last.set_index("channel")["share"]
    assert float(shares.get("type_B", 0.0)) > 0.0
    canonical = events[["user_id", "timestamp", "channel", "conversion", "conversion_value"]]
    report = MTAAudit(data=canonical).run(
        attribution_models=["first_touch", "last_touch", "linear"],
        checks=["model_disagreement"],
    )
    assert report.results[0].name == "model_disagreement"


def test_synthesizer_round_trip_and_audit() -> None:
    generated = simulate_synthesizer_events(120, random_state=11)
    mapped = transform_synthesizer_touchpoints(
        generated[["uid", "touch_sequence", "channel", "conversion", "monetary_value"]]
    )
    assert mapped["conversion"].sum() == generated["conversion"].sum()
    assert mapped["user_id"].nunique() == 120
    report = MTAAudit(data=mapped).run(
        attribution_models=["first_touch", "last_touch", "linear"],
        checks=["data_quality"],
    )
    assert report.results[0].status.value != "error"


def test_ignazio_marks_one_conversion_per_converting_journey() -> None:
    raw = pd.read_csv(FIXTURES / "ignazio_mini.csv")
    events = transform_ignazio_touchpoints(raw)
    assert int(events["conversion"].sum()) == 1
    converting = events[events["conversion"]].iloc[0]
    assert converting["channel"] == "email_nurture"
    assert events.loc[events["user_id"].eq("2"), "conversion"].sum() == 0


def test_triangulation_adapter_keeps_user_level_flags() -> None:
    raw = pd.read_csv(FIXTURES / "triangulation_mini.csv")
    events = transform_triangulation_mta(raw)
    assert events["conversion"].tolist() == [False, False, True]
    assert events["channel"].tolist() == ["display", "paid_search", "paid_search"]


def test_missing_external_file_is_explicit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("MTA_AUDIT_IGNAZIO_PATH", raising=False)
    monkeypatch.setenv("MTA_AUDIT_CACHE_DIR", str(tmp_path / "cache"))
    with pytest.raises(DatasetNotAvailableError, match="IgnazioDS"):
        load_ignazio_attribution(download=False)


def test_external_transformers_reject_incomplete_frames() -> None:
    empty = pd.DataFrame({"channel": ["search"]})
    with pytest.raises(SchemaError, match="Synthesizer"):
        transform_synthesizer_touchpoints(empty)
    with pytest.raises(SchemaError, match="Ignazio"):
        transform_ignazio_touchpoints(empty)
    with pytest.raises(SchemaError, match="Triangulation"):
        transform_triangulation_mta(empty)
