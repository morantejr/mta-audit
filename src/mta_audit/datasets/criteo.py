"""Adapter for the public Criteo Attribution Modeling dataset.

The dataset is a 30-day sample of display-ad impressions (Diemert et al., 2017).
It is licensed CC-BY-NC-SA-4.0 and is **not** redistributed with this package.

Important field semantics (do not treat these as mta-audit defaults):

- Each row is an *impression*, not a conversion event.
- ``conversion`` is 1 if *that impression* was followed by a conversion within
  30 days. Every impression on a converting timeline typically carries
  ``conversion=1``. Using the column as ``conversion_col`` would treat almost
  every converting-user impression as a conversion event.
- ``conversion_timestamp`` / ``conversion_id`` identify the actual outcome.
- ``campaign`` is an anonymized campaign identifier. Criteo does not publish a
  named media-channel taxonomy, so this adapter maps campaign → ``channel``.
  Results are campaign-level attribution, not search/social/display labels.
- ``timestamp`` is seconds from the first impression in the file (starting at 0),
  not a wall-clock Unix time. We convert it with ``unit='s'`` so day-based
  windows remain valid relative durations.
- ``attribution`` is Criteo's own last-click-style label and is preserved but
  never used as this package's conversion indicator.
- ``cost`` / ``cpo`` are transformed media/order values, not conversion value.

The adapter therefore constructs event-level conversions by marking, for each
``(uid, conversion_id)``, the last impression at or before
``conversion_timestamp``. The true conversion time is stored in
``conversion_timestamp`` so lookback, windows, and delayed-converter checks use
outcome time rather than impression time.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterable
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

import pandas as pd

from ..schema import SchemaError

logger = logging.getLogger(__name__)

CRITEO_DATASET_URL = (
    "https://huggingface.co/datasets/criteo/criteo-attribution-dataset"
    "/resolve/main/criteo_attribution_dataset.tsv.gz"
)
CRITEO_FILENAME = "criteo_attribution_dataset.tsv.gz"
CRITEO_COLUMNS: tuple[str, ...] = (
    "timestamp",
    "uid",
    "campaign",
    "conversion",
    "conversion_timestamp",
    "conversion_id",
    "attribution",
    "click",
    "click_pos",
    "click_nb",
    "cost",
    "cpo",
    "time_since_last_click",
    "cat1",
    "cat2",
    "cat3",
    "cat4",
    "cat5",
    "cat6",
    "cat7",
    "cat8",
    "cat9",
)
REQUIRED_CRITEO_COLUMNS: tuple[str, ...] = (
    "timestamp",
    "uid",
    "campaign",
    "conversion",
    "conversion_timestamp",
    "conversion_id",
)

_INSTRUCTIONS = f"""Obtain the public Criteo Attribution Modeling dataset yourself
under its current terms (CC-BY-NC-SA-4.0) and place the file locally:

  {CRITEO_DATASET_URL}

Save it as ./data/{CRITEO_FILENAME} or pass path=... to load_criteo_attribution.
This package does not bundle the file. Optional download:

  load_criteo_attribution(download=True)
"""


class DatasetNotAvailableError(FileNotFoundError):
    """Raised when a public dataset file is missing and download was not requested."""


class CriteoAttributionAdapter:
    """Load and map Criteo impressions into the mta-audit event schema."""

    def __init__(
        self,
        path: str | Path | None = None,
        *,
        download: bool = False,
        sample_users: int | None = None,
        random_state: int | None = 42,
        cache_dir: str | Path | None = None,
    ) -> None:
        self.path = Path(path) if path is not None else None
        self.download = download
        self.sample_users = sample_users
        self.random_state = random_state
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None

    def load(self) -> pd.DataFrame:
        """Return mapped events ready for ``MTAAudit`` / ``build_journeys``."""

        return load_criteo_attribution(
            path=self.path,
            download=self.download,
            sample_users=self.sample_users,
            random_state=self.random_state,
            cache_dir=self.cache_dir,
        )


def load_criteo_attribution(
    path: str | Path | None = None,
    *,
    download: bool = False,
    sample_users: int | None = None,
    random_state: int | None = 42,
    cache_dir: str | Path | None = None,
    nrows: int | None = None,
) -> pd.DataFrame:
    """Load Criteo impressions, validate columns, and map to standard events.

    Parameters
    ----------
    path:
        Local ``.tsv`` or ``.tsv.gz`` path. If omitted, the loader searches
        ``MTA_AUDIT_CRITEO_PATH``, ``./data/criteo_attribution_dataset.tsv.gz``,
        and ``~/.cache/mta-audit/criteo_attribution_dataset.tsv.gz``.
    download:
        When True and the file is absent, download it from Hugging Face into the
        cache directory. Disabled by default so unit tests never hit the network.
    sample_users:
        If set, keep a deterministic sample of users. This is **explicit**
        representative sampling, not silent downsampling. ``None`` loads every
        row that was read.
    nrows:
        Optional raw-row cap applied before mapping. Intended for smoke tests.
        Documented separately from ``sample_users`` because it is not a
        user-level sample.
    """

    source = resolve_criteo_path(path, download=download, cache_dir=cache_dir)
    raw = read_criteo_file(source, nrows=nrows)
    if sample_users is not None:
        raw = _sample_users(raw, sample_users, random_state)
        logger.warning(
            "Using an explicit representative sample of %s users (random_state=%s). "
            "Pass sample_users=None to keep every loaded user.",
            sample_users,
            random_state,
        )
    return transform_criteo_events(raw)


def resolve_criteo_path(
    path: str | Path | None = None,
    *,
    download: bool = False,
    cache_dir: str | Path | None = None,
) -> Path:
    """Locate a local Criteo file, optionally downloading it."""

    candidates: list[Path] = []
    if path is not None:
        candidates.append(Path(path))
    env_path = os.environ.get("MTA_AUDIT_CRITEO_PATH")
    if env_path:
        candidates.append(Path(env_path))
    cache = Path(cache_dir) if cache_dir is not None else default_cache_dir()
    candidates.extend(
        [
            Path("data") / CRITEO_FILENAME,
            cache / CRITEO_FILENAME,
        ]
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    if download:
        destination = Path(path) if path is not None else cache / CRITEO_FILENAME
        download_criteo_dataset(destination)
        return destination
    raise DatasetNotAvailableError(_INSTRUCTIONS)


def default_cache_dir() -> Path:
    """Return the default on-disk cache directory."""

    override = os.environ.get("MTA_AUDIT_CACHE_DIR")
    if override:
        return Path(override)
    return Path.home() / ".cache" / "mta-audit"


def download_criteo_dataset(destination: Path, url: str = CRITEO_DATASET_URL) -> Path:
    """Download the public TSV into ``destination`` if it does not already exist."""

    destination = Path(destination)
    if destination.is_file():
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading Criteo attribution dataset to %s", destination)
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
            f"Failed to download the Criteo dataset from {url}: {exc}\n{_INSTRUCTIONS}"
        ) from exc
    tmp.replace(destination)
    return destination


def read_criteo_file(path: str | Path, *, nrows: int | None = None) -> pd.DataFrame:
    """Read a Criteo TSV and validate the published column set."""

    frame = pd.read_csv(path, sep="\t", nrows=nrows)
    missing = [column for column in REQUIRED_CRITEO_COLUMNS if column not in frame.columns]
    if missing:
        raise SchemaError(f"Criteo file is missing required columns: {missing}")
    return frame


def transform_criteo_events(raw: pd.DataFrame) -> pd.DataFrame:
    """Map Criteo impressions onto the package event schema without extra credit rows."""

    missing = [column for column in REQUIRED_CRITEO_COLUMNS if column not in raw.columns]
    if missing:
        raise SchemaError(f"Criteo frame is missing required columns: {missing}")
    events = raw.copy()
    events["timestamp"] = _relative_seconds_to_timestamp(events["timestamp"])
    conversion_ts = events["conversion_timestamp"].where(events["conversion_timestamp"] >= 0)
    events["conversion_timestamp"] = _relative_seconds_to_timestamp(conversion_ts)
    events["channel"] = events["campaign"].astype(str)
    events["user_id"] = events["uid"]
    events["event_conversion"] = False
    converting = events["conversion_id"].ne(-1) & events["conversion"].eq(1)
    eligible = events.loc[
        converting
        & events["conversion_timestamp"].notna()
        & (events["timestamp"] <= events["conversion_timestamp"])
    ]
    if not eligible.empty:
        marked = eligible.groupby(["uid", "conversion_id"], sort=False)["timestamp"].idxmax()
        events.loc[marked, "event_conversion"] = True
    events["conversion"] = events["event_conversion"]
    events = events.drop(columns=["event_conversion"])
    if "click" in events.columns:
        events["click"] = events["click"].astype(bool)
    return events.reset_index(drop=True)


def _relative_seconds_to_timestamp(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    return pd.to_datetime(numeric, unit="s", utc=True, origin="unix")


def _sample_users(
    raw: pd.DataFrame, sample_users: int, random_state: int | None
) -> pd.DataFrame:
    if sample_users <= 0:
        raise ValueError("sample_users must be a positive integer.")
    users = raw["uid"].drop_duplicates()
    if sample_users >= len(users):
        return raw
    selected = users.sample(n=sample_users, random_state=random_state)
    return raw.loc[raw["uid"].isin(selected)].copy()


def criteo_column_mapping() -> dict[str, str]:
    """Return the mapped-column configuration after ``transform_criteo_events``."""

    return {
        "user_id": "user_id",
        "timestamp": "timestamp",
        "channel": "channel",
        "conversion": "conversion",
        "conversion_timestamp": "conversion_timestamp",
        "campaign_id": "campaign",
        "click": "click",
        "cost": "cost",
        "conversion_id": "conversion_id",
    }


def describe_sampling(sample_users: int | None, loaded_users: int, total_hint: str) -> str:
    """Human-readable note used by benchmarks and the notebook."""

    if sample_users is None:
        return f"Loaded all {loaded_users} users from the local Criteo extract ({total_hint})."
    return (
        f"Explicit representative sample of {loaded_users} users "
        f"(requested sample_users={sample_users}) from the Criteo extract ({total_hint}). "
        "This is not silent downsampling."
    )


def iter_known_locations() -> Iterable[Path]:
    """Yield default search locations without downloading."""

    env_path = os.environ.get("MTA_AUDIT_CRITEO_PATH")
    if env_path:
        yield Path(env_path)
    yield Path("data") / CRITEO_FILENAME
    yield default_cache_dir() / CRITEO_FILENAME
