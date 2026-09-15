from pathlib import Path

import pandas as pd
import pytest

from mta_audit import JourneyBuilder, build_journeys
from mta_audit.attribution import LinearAttribution
from mta_audit.datasets import (
    CriteoAttributionAdapter,
    DatasetNotAvailableError,
    load_criteo_attribution,
    transform_criteo_events,
)
from mta_audit.datasets.criteo import read_criteo_file, resolve_criteo_path
from mta_audit.datasets.evaluation import attribution_share_drift, rank_correlation

FIXTURE = Path(__file__).parent / "fixtures" / "criteo_mini.tsv"


def test_transform_marks_last_impression_before_conversion_not_every_flagged_row() -> None:
    raw = read_criteo_file(FIXTURE)
    events = transform_criteo_events(raw)

    converter = events[events["user_id"] == 1]
    assert int(converter["conversion"].sum()) == 1
    marked = converter.loc[converter["conversion"]].iloc[0]
    assert str(marked["channel"]) == "200"
    assert marked["timestamp"] < marked["conversion_timestamp"]
    later = converter.loc[converter["channel"].astype(str).eq("300"), "conversion"].iloc[0]
    assert not bool(later)


def test_loader_and_adapter_build_converter_and_non_converter_journeys() -> None:
    events = CriteoAttributionAdapter(FIXTURE).load()
    journeys = JourneyBuilder(
        events,
        user_col="user_id",
        timestamp_col="timestamp",
        channel_col="channel",
        conversion_col="conversion",
        lookback_window="30D",
    ).build()
    converters = [journey for journey in journeys if journey.converted]
    non_converters = [journey for journey in journeys if not journey.converted]
    assert len(converters) == 2
    assert len(non_converters) == 1
    first_converter = next(journey for journey in converters if journey.user_id == 1)
    assert first_converter.channels == ("100", "200")
    assert non_converters[0].channels == ("100", "400")
    for journey in journeys:
        stamps = [touch.timestamp for touch in journey.touchpoints]
        assert stamps == sorted(stamps)


def test_conversion_timestamp_is_used_as_journey_endpoint() -> None:
    events = load_criteo_attribution(FIXTURE)
    conversion_time = events.loc[
        (events["user_id"] == 1) & events["conversion"], "conversion_timestamp"
    ].iloc[0]
    converter = next(
        journey for journey in build_journeys(events) if journey.user_id == 1 and journey.converted
    )
    assert converter.conversion_time == conversion_time


def test_missing_dataset_fails_gracefully(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MTA_AUDIT_CRITEO_PATH", raising=False)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MTA_AUDIT_CACHE_DIR", str(tmp_path / "cache"))
    with pytest.raises(DatasetNotAvailableError, match="CC-BY-NC-SA"):
        resolve_criteo_path(download=False)


def test_attribution_drift_helpers_are_symmetric() -> None:
    events = load_criteo_attribution(FIXTURE)
    result = LinearAttribution().attribute(build_journeys(events))
    assert attribution_share_drift(result, result) == pytest.approx(0.0)
    assert rank_correlation(result, result) == pytest.approx(1.0)
    assert isinstance(pd.DataFrame(result.data), pd.DataFrame)
