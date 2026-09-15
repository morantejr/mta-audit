"""Independent synthetic MTA fixtures inspired by public research generators.

None of the upstream generators are vendored:

- JD MTA simulation (Du et al., 2019) is GPL-2.0 TensorFlow + a 49 MB tfrecord.
  ``simulate_jd_mta_events`` follows the *published sequential DGP structure*
  at test scale. It is not a copy of ``generate_simulation_data.py``.
- Criteo Research robust-label experiments (Bompaire, Heymann, Désir, 2020)
  live in an Apache-2.0 notebook. ``simulate_criteo_label_events`` follows the
  paper's sequential display / leave / last-touch-label setup.
- Multi-Touch Synthesizer (PubliusV) has no license; we reimplement the
  published Poisson + weighted-channel process and accept their CSV schema.
- IgnazioDS marketing-attribution and de-lazurenko triangulation publish
  event CSVs. Adapters map those files; generators are not copied. The
  triangulation weekly/geo tables stay out of this module (MMM / lift).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

from ..schema import SchemaError
from .criteo import DatasetNotAvailableError, default_cache_dir

logger = logging.getLogger(__name__)

IGNATIO_TOUCHPOINTS_URL = (
    "https://raw.githubusercontent.com/IgnazioDS/marketing-attribution/main/data/touchpoints.csv"
)
TRIANGULATION_MTA_URL = (
    "https://raw.githubusercontent.com/de-lazurenko/"
    "marketing_measurement_triangulation/main/data/processed/df_mta.csv"
)

_SYNTHESIZER_DEFAULTS: tuple[tuple[str, float, float, float, float], ...] = (
    ("Paid Social", 0.02, 0.20, 95.0, 5.0),
    ("Paid Search", 0.02, 0.10, 90.0, 10.0),
    ("Display Network", 0.025, 0.30, 100.0, 30.0),
    ("Affiliate", 0.04, 0.30, 120.0, 15.0),
    ("Email", 0.05, 0.10, 130.0, 10.0),
)


def simulate_jd_mta_events(
    n_users: int = 200,
    *,
    n_days: int = 5,
    n_brands: int = 2,
    n_positions: int = 2,
    avg_impressions: float = 1.5,
    random_state: int = 42,
) -> pd.DataFrame:
    """Simulate sequential brand/position exposures with recurrent conversion.

    Structure matches Du, Zhong, Nair, Cui, and Shou (2019): Poisson exposures,
    a recurrent attractiveness term, and Bernoulli conversions. Coefficients
    are test-scale so a few hundred users produce converters and NULL paths;
    they are not the paper's 100,000-user TensorFlow run.
    """

    if min(n_users, n_days, n_brands, n_positions) < 1:
        raise ValueError("n_users, n_days, n_brands, and n_positions must be >= 1")
    rng = np.random.default_rng(random_state)
    origin = pd.Timestamp("2018-01-01", tz="UTC")
    decay = 0.5
    alpha0 = 0.0
    alpha1 = rng.uniform(0.15, 0.55, size=(n_brands, n_positions))
    alpha2 = rng.uniform(0.05, 0.25, size=n_brands)
    beta0 = -4.2
    beta1 = rng.uniform(0.05, 0.25, size=(n_brands, n_positions))
    beta2 = 0.15
    beta3 = rng.uniform(0.02, 0.08, size=n_brands)
    prices = rng.lognormal(mean=0.0, sigma=0.15, size=(n_days, n_brands))
    rows: list[dict[str, object]] = []
    for user in range(n_users):
        exposures = rng.poisson(avg_impressions, size=(n_days, n_brands, n_positions))
        trait = float(rng.uniform())
        hidden = np.zeros((n_days + 1, n_brands))
        hidden[0] = rng.normal(size=n_brands)
        for day in range(n_days):
            for brand in range(n_brands):
                drive = (
                    alpha0
                    + float(exposures[day, brand] @ alpha1[brand])
                    + float(prices[day, brand] * alpha2[brand])
                )
                hidden[day + 1, brand] = (
                    decay * _sigmoid(drive) + (1.0 - decay) * hidden[day, brand]
                )
                utility = (
                    beta0 * trait
                    + float(exposures[day, brand] @ beta1[brand])
                    + float(hidden[day + 1, brand] * beta2)
                    + float(prices[day, brand] * beta3[brand])
                )
                converted = bool(rng.random() < _sigmoid(utility))
                counts = exposures[day, brand]
                if converted and int(counts.sum()) == 0:
                    counts = counts.copy()
                    counts[0] = 1
                step = 0
                total = int(counts.sum())
                for position, count in enumerate(counts.tolist()):
                    for _ in range(int(count)):
                        step += 1
                        stamp = origin + pd.Timedelta(days=day) + pd.Timedelta(hours=step)
                        is_conversion = converted and step == total
                        rows.append(
                            {
                                "user_id": f"jd_{user}",
                                "timestamp": stamp,
                                "channel": f"brand{brand}_pos{position}",
                                "conversion": is_conversion,
                                "conversion_value": 1.0 if is_conversion else 0.0,
                                "campaign_id": f"brand{brand}",
                            }
                        )
    return pd.DataFrame(rows)


def simulate_criteo_label_events(
    n_users: int = 400,
    *,
    alpha_a: float = 0.20,
    alpha_b: float = 0.0,
    leave_prob: float = 0.25,
    p_type_a: float = 0.5,
    max_steps: int = 16,
    random_state: int = 42,
) -> pd.DataFrame:
    """Simulate Criteo-style last-touch label mismatch on sequential displays.

    Type A can trigger a conversion; type B does so only at ``alpha_b`` (0 in
    the paper's Setup 2). The user leaves with probability ``leave_prob`` after
    each display. Outcome credit is attached to the *last* display, which is
    the last-touch labeling problem studied by Bompaire et al. (2020).
    """

    if n_users < 1 or max_steps < 1:
        raise ValueError("n_users and max_steps must be >= 1")
    rng = np.random.default_rng(random_state)
    origin = pd.Timestamp("2020-01-01", tz="UTC")
    rows: list[dict[str, object]] = []
    for user in range(n_users):
        triggered = False
        true_channel: str | None = None
        steps: list[str] = []
        for _step in range(max_steps):
            channel = "type_A" if rng.random() < p_type_a else "type_B"
            steps.append(channel)
            rate = alpha_a if channel == "type_A" else alpha_b
            if rng.random() < rate:
                triggered = True
                if true_channel is None:
                    true_channel = channel
            if rng.random() < leave_prob:
                break
        for index, channel in enumerate(steps):
            is_last = index == len(steps) - 1
            is_conversion = triggered and is_last
            rows.append(
                {
                    "user_id": f"lab_{user}",
                    "timestamp": origin + pd.Timedelta(hours=index),
                    "channel": channel,
                    "conversion": is_conversion,
                    "conversion_value": 1.0 if is_conversion else 0.0,
                    "true_trigger_channel": true_channel if is_conversion else None,
                }
            )
    return pd.DataFrame(rows)


def simulate_synthesizer_events(
    n_users: int = 300,
    *,
    touchpoint_lambda: float = 4.0,
    random_state: int = 42,
) -> pd.DataFrame:
    """Simulate Poisson-length journeys with weighted channels and last-touch CVR.

    Compatible with the Multi-Touch Synthesizer export schema
    (``uid``, ``touch_sequence``, ``channel``, ``conversion``, ``monetary_value``).
    """

    if n_users < 1:
        raise ValueError("n_users must be >= 1")
    rng = np.random.default_rng(random_state)
    names = [row[0] for row in _SYNTHESIZER_DEFAULTS]
    cvrs = np.array([row[1] for row in _SYNTHESIZER_DEFAULTS], dtype=float)
    weights = np.array([row[2] for row in _SYNTHESIZER_DEFAULTS], dtype=float)
    weights = weights / weights.sum()
    aovs = np.array([row[3] for row in _SYNTHESIZER_DEFAULTS], dtype=float)
    stdevs = np.array([row[4] for row in _SYNTHESIZER_DEFAULTS], dtype=float)
    origin = pd.Timestamp("2024-01-01", tz="UTC")
    rows: list[dict[str, object]] = []
    for user in range(n_users):
        n_touches = max(1, int(rng.poisson(touchpoint_lambda)))
        indices = rng.choice(len(names), size=n_touches, p=weights)
        cumulative = float(np.clip(cvrs[indices].sum(), 0.0, 0.95))
        converts = bool(rng.random() < cumulative)
        value = 0.0
        if converts:
            last = int(indices[-1])
            value = max(1.0, float(rng.normal(aovs[last], stdevs[last])))
        for step, channel_index in enumerate(indices.tolist()):
            is_conversion = converts and step == n_touches - 1
            rows.append(
                {
                    "uid": user,
                    "touch_sequence": step + 1,
                    "channel": names[channel_index],
                    "conversion": int(is_conversion),
                    "monetary_value": value if is_conversion else 0.0,
                    "user_id": f"syn_{user}",
                    "timestamp": origin + pd.Timedelta(days=step),
                    "conversion_value": value if is_conversion else 0.0,
                }
            )
    frame = pd.DataFrame(rows)
    frame["conversion"] = frame["conversion"].astype(bool)
    return frame


def transform_synthesizer_touchpoints(touchpoints: pd.DataFrame) -> pd.DataFrame:
    """Map a Multi-Touch Synthesizer CSV onto the package event schema."""

    required = {"uid", "touch_sequence", "channel", "conversion"}
    missing = sorted(required - set(touchpoints.columns))
    if missing:
        raise SchemaError(f"Synthesizer touchpoints missing columns: {missing}")
    origin = pd.Timestamp("2024-01-01", tz="UTC")
    sequence = pd.to_numeric(touchpoints["touch_sequence"], errors="coerce").fillna(1)
    converted = _as_bool(touchpoints["conversion"])
    value = (
        pd.to_numeric(touchpoints["monetary_value"], errors="coerce").fillna(0.0)
        if "monetary_value" in touchpoints.columns
        else converted.astype(float)
    )
    events = pd.DataFrame(
        {
            "user_id": touchpoints["uid"].astype(str),
            "timestamp": origin + pd.to_timedelta(sequence - 1, unit="D"),
            "channel": touchpoints["channel"].astype(str),
            "conversion": converted.to_numpy(dtype=bool),
            "conversion_value": value.where(converted, 0.0),
        }
    )
    return events.reset_index(drop=True)


def transform_ignazio_touchpoints(touchpoints: pd.DataFrame) -> pd.DataFrame:
    """Map IgnazioDS synthetic journeys onto events.

    ``converted`` is journey-level and is repeated on every row of a converting
    path. Conversion is marked on the last touch of those journeys.
    """

    required = {"journey_id", "channel", "touch_timestamp", "converted"}
    missing = sorted(required - set(touchpoints.columns))
    if missing:
        raise SchemaError(f"Ignazio touchpoints missing columns: {missing}")
    frame = touchpoints.copy()
    frame["timestamp"] = pd.to_datetime(frame["touch_timestamp"], utc=True)
    if "touch_position" in frame.columns:
        sort_cols = ["journey_id", "touch_position", "timestamp"]
    else:
        sort_cols = ["journey_id", "timestamp"]
    frame = frame.sort_values(sort_cols)
    journey_converted = (
        _as_bool(frame["converted"]).groupby(frame["journey_id"]).transform("max").astype(bool)
    )
    last_index = frame.groupby("journey_id", sort=False).tail(1).index
    is_conversion = pd.Series(False, index=frame.index)
    is_conversion.loc[last_index] = journey_converted.loc[last_index]
    value = (
        pd.to_numeric(frame["revenue"], errors="coerce").fillna(0.0)
        if "revenue" in frame.columns
        else is_conversion.astype(float)
    )
    events = pd.DataFrame(
        {
            "user_id": frame["journey_id"].astype(str),
            "timestamp": frame["timestamp"],
            "channel": frame["channel"].astype(str),
            "conversion": is_conversion.to_numpy(dtype=bool),
            "conversion_value": value.where(is_conversion, 0.0).to_numpy(),
        }
    )
    if "conversion_timestamp" in frame.columns:
        events["conversion_timestamp"] = pd.to_datetime(
            frame["conversion_timestamp"], utc=True, errors="coerce"
        )
    if "cost" in frame.columns:
        events["cost"] = pd.to_numeric(frame["cost"], errors="coerce")
    return events.reset_index(drop=True)


def transform_triangulation_mta(touchpoints: pd.DataFrame) -> pd.DataFrame:
    """Map the triangulation project's user-level MTA table onto events.

    Weekly MMM and geo-holdout files are ignored; they are not journey MTA.
    """

    required = {"user_id", "channel", "conversion"}
    missing = sorted(required - set(touchpoints.columns))
    if missing:
        raise SchemaError(f"Triangulation MTA table missing columns: {missing}")
    stamp_col = "touchpoint_date" if "touchpoint_date" in touchpoints.columns else "timestamp"
    if stamp_col not in touchpoints.columns:
        raise SchemaError("Triangulation MTA table missing timestamp column")
    converted = _as_bool(touchpoints["conversion"])
    events = pd.DataFrame(
        {
            "user_id": touchpoints["user_id"].astype(str),
            "timestamp": pd.to_datetime(touchpoints[stamp_col], utc=True),
            "channel": touchpoints["channel"].astype(str),
            "conversion": converted.to_numpy(dtype=bool),
            "conversion_value": converted.astype(float),
        }
    )
    return events.reset_index(drop=True)


def load_ignazio_attribution(
    path: str | Path | None = None,
    *,
    download: bool = False,
) -> pd.DataFrame:
    """Load the public IgnazioDS synthetic touchpoints CSV if present."""

    file = _resolve_external_file(
        path,
        "ignazio_touchpoints.csv",
        IGNATIO_TOUCHPOINTS_URL,
        download=download,
        env_key="MTA_AUDIT_IGNAZIO_PATH",
        hint="IgnazioDS/marketing-attribution data/touchpoints.csv",
    )
    return transform_ignazio_touchpoints(pd.read_csv(file))


def load_triangulation_mta(
    path: str | Path | None = None,
    *,
    download: bool = False,
) -> pd.DataFrame:
    """Load the triangulation project's df_mta.csv if present."""

    file = _resolve_external_file(
        path,
        "triangulation_df_mta.csv",
        TRIANGULATION_MTA_URL,
        download=download,
        env_key="MTA_AUDIT_TRIANGULATION_PATH",
        hint="de-lazurenko/marketing_measurement_triangulation data/processed/df_mta.csv",
    )
    return transform_triangulation_mta(pd.read_csv(file))


def _sigmoid(value: float) -> float:
    clipped = min(40.0, max(-40.0, float(value)))
    return 1.0 / (1.0 + np.exp(-clipped))


def _as_bool(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().all():
        return numeric.fillna(0).ne(0)
    lowered = series.astype(str).str.strip().str.lower()
    return lowered.isin({"1", "true", "yes"})


def _resolve_external_file(
    path: str | Path | None,
    filename: str,
    url: str,
    *,
    download: bool,
    env_key: str,
    hint: str,
) -> Path:
    instructions = (
        f"Obtain {hint} yourself. This package does not bundle the file.\n"
        f"  {url}\n"
        f"Optional: pass download=True."
    )
    if path is not None:
        resolved = Path(path)
        if resolved.is_file():
            return resolved
        raise DatasetNotAvailableError(f"File not found: {resolved}\n{instructions}")
    candidates: list[Path] = []
    if os.environ.get(env_key):
        candidates.append(Path(os.environ[env_key]))
    candidates.append(Path("data") / filename)
    candidates.append(default_cache_dir() / filename)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    destination = default_cache_dir() / filename
    if download:
        return _download(destination, url, instructions)
    raise DatasetNotAvailableError(instructions)


def _download(destination: Path, url: str, instructions: str) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading external synthetic file to %s", destination)
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
        raise DatasetNotAvailableError(f"Failed to download {url}: {exc}\n{instructions}") from exc
    tmp.replace(destination)
    return destination
