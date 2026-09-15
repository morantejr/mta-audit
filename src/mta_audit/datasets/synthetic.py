"""Adapter for the public Hugging Face synthetic MTA benchmark.

Kaggle currently has no licensed, event-level multi-touch path dataset with
converters and non-converters comparable to Criteo. This CC-BY-4.0 synthetic
benchmark (lucianfialho/synthetic-attribution-benchmark) provides named channels
and ground-truth Shapley shares, so Markov identification and model disagreement
can be tested without warehouse data.

It is not redistributed with the package.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

import pandas as pd

from ..schema import SchemaError
from .criteo import DatasetNotAvailableError, default_cache_dir

logger = logging.getLogger(__name__)

SYNTHETIC_REPO = "https://huggingface.co/datasets/lucianfialho/synthetic-attribution-benchmark"
SYNTHETIC_TOUCHPOINTS_URL = f"{SYNTHETIC_REPO}/resolve/main/touchpoints.parquet"
SYNTHETIC_JOURNEYS_URL = f"{SYNTHETIC_REPO}/resolve/main/journeys.parquet"

_INSTRUCTIONS = f"""Obtain the public synthetic MTA benchmark yourself.

  {SYNTHETIC_REPO}

License: CC BY 4.0. Files are not stored in git.

  load_synthetic_attribution(download=True)
"""


def load_synthetic_attribution(
    touchpoints_path: str | Path | None = None,
    journeys_path: str | Path | None = None,
    *,
    download: bool = False,
) -> pd.DataFrame:
    """Load synthetic touchpoints + journey outcomes as package events."""

    touch_file = _resolve_file(
        touchpoints_path,
        "synthetic_touchpoints.parquet",
        SYNTHETIC_TOUCHPOINTS_URL,
        download=download,
    )
    journey_file = _resolve_file(
        journeys_path,
        "synthetic_journeys.parquet",
        SYNTHETIC_JOURNEYS_URL,
        download=download,
    )
    touchpoints = _read_table(touch_file)
    journeys = _read_table(journey_file)
    return transform_synthetic_attribution(touchpoints, journeys)


def transform_synthetic_attribution(
    touchpoints: pd.DataFrame,
    journeys: pd.DataFrame,
) -> pd.DataFrame:
    """Map synthetic journey files onto the package event schema."""

    required_touch = {"journey_id", "step", "channel"}
    missing_touch = sorted(required_touch - set(touchpoints.columns))
    if missing_touch:
        raise SchemaError(f"Synthetic touchpoints missing columns: {missing_touch}")
    required_journey = {"journey_id", "converted"}
    missing_journey = sorted(required_journey - set(journeys.columns))
    if missing_journey:
        raise SchemaError(f"Synthetic journeys missing columns: {missing_journey}")
    merged = touchpoints.merge(
        journeys[["journey_id", "converted"]],
        on="journey_id",
        how="left",
    )
    if "t_hours" in merged.columns:
        origin = pd.Timestamp("2020-01-01", tz="UTC")
        merged["timestamp"] = origin + pd.to_timedelta(merged["t_hours"], unit="h")
    elif "timestamp" not in merged.columns:
        merged["timestamp"] = pd.Timestamp("2020-01-01", tz="UTC") + pd.to_timedelta(
            pd.to_numeric(merged["step"], errors="coerce").fillna(0), unit="D"
        )
    max_step = merged.groupby("journey_id")["step"].transform("max")
    converted = pd.to_numeric(merged["converted"], errors="coerce").fillna(0).ne(0)
    is_conversion = converted & merged["step"].eq(max_step)
    events = pd.DataFrame(
        {
            "user_id": merged["journey_id"].astype(str),
            "timestamp": pd.to_datetime(merged["timestamp"], utc=True),
            "channel": merged["channel"].astype(str),
            "conversion": is_conversion.to_numpy(dtype=bool),
            "conversion_value": is_conversion.astype(float),
        }
    )
    if "cost" in merged.columns:
        events["cost"] = pd.to_numeric(merged["cost"], errors="coerce")
    return events.reset_index(drop=True)


def _resolve_file(
    path: str | Path | None,
    filename: str,
    url: str,
    *,
    download: bool,
) -> Path:
    if path is not None:
        resolved = Path(path)
        if resolved.is_file():
            return resolved
        raise DatasetNotAvailableError(f"Synthetic file not found: {resolved}\n{_INSTRUCTIONS}")
    env_key = "MTA_AUDIT_SYNTHETIC_DIR"
    candidates = []
    if os.environ.get(env_key):
        candidates.append(Path(os.environ[env_key]) / filename)
    candidates.append(Path("data") / filename)
    candidates.append(default_cache_dir() / filename)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    destination = default_cache_dir() / filename
    if download:
        return _download(destination, url)
    raise DatasetNotAvailableError(_INSTRUCTIONS)


def _download(destination: Path, url: str) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading synthetic attribution file to %s", destination)
    request = Request(url, headers={"User-Agent": "mta-audit/0.1"})
    tmp = destination.with_suffix(destination.suffix + ".part")
    try:
        with urlopen(request, timeout=120) as response, tmp.open("wb") as handle:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)
    except URLError as exc:
        if tmp.exists():
            tmp.unlink()
        raise DatasetNotAvailableError(
            f"Failed to download {url}: {exc}\n{_INSTRUCTIONS}"
        ) from exc
    tmp.replace(destination)
    return destination


def _read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    try:
        return pd.read_parquet(path)
    except ImportError as exc:
        raise DatasetNotAvailableError(
            "Reading parquet requires pyarrow. Install mta-audit[parquet] "
            "or pass CSV paths to transform_synthetic_attribution."
        ) from exc
