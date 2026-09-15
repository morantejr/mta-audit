"""Local, journey-preserving sampling helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .schema import ColumnMapping


def sample_for_audit(
    data: pd.DataFrame,
    *,
    max_journeys: int = 100_000,
    stratify_by: str | None = "conversion",
    random_state: int | None = 42,
    columns: ColumnMapping | dict[str, str] | None = None,
) -> pd.DataFrame:
    """Sample complete user histories for a local audit.

    ``max_journeys`` is an upper bound on selected user histories. Users with
    multiple conversions may still produce multiple journeys after construction.
    Sampling metadata is stored in ``DataFrame.attrs`` and propagated to reports.
    """

    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame.")
    if isinstance(max_journeys, bool) or max_journeys < 1:
        raise ValueError("max_journeys must be a positive integer.")
    if stratify_by not in {None, "conversion"}:
        raise ValueError("stratify_by must be 'conversion' or None.")
    mapping = ColumnMapping.from_value(columns)
    required = {mapping.user_id, mapping.conversion}
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"Missing required columns for sampling: {missing}")

    users = pd.Index(data[mapping.user_id].drop_duplicates())
    if len(users) <= max_journeys:
        result = data.copy()
        result.attrs["mta_audit_sample"] = {
            "sampled": False,
            "input_rows": len(data),
            "output_rows": len(result),
            "input_users": len(users),
            "output_users": len(users),
            "max_journeys": max_journeys,
            "stratify_by": stratify_by,
            "random_state": random_state,
        }
        return result

    rng = np.random.default_rng(random_state)
    if stratify_by == "conversion":
        conversion = _truthy(data[mapping.conversion])
        converter_users = pd.Index(data.loc[conversion, mapping.user_id].unique())
        non_converter_users = users.difference(converter_users, sort=False)
        converter_target = round(max_journeys * len(converter_users) / len(users))
        converter_target = min(len(converter_users), max(1, converter_target))
        non_converter_target = min(
            len(non_converter_users), max_journeys - converter_target
        )
        if converter_target + non_converter_target < max_journeys:
            converter_target = min(
                len(converter_users), max_journeys - non_converter_target
            )
        selected = np.concatenate(
            [
                rng.choice(converter_users.to_numpy(), converter_target, replace=False),
                rng.choice(
                    non_converter_users.to_numpy(), non_converter_target, replace=False
                ),
            ]
        )
    else:
        selected = rng.choice(users.to_numpy(), max_journeys, replace=False)

    result = data.loc[data[mapping.user_id].isin(set(selected))].copy()
    result.attrs["mta_audit_sample"] = {
        "sampled": True,
        "input_rows": len(data),
        "output_rows": len(result),
        "input_users": len(users),
        "output_users": len(pd.unique(result[mapping.user_id])),
        "max_journeys": max_journeys,
        "stratify_by": stratify_by,
        "random_state": random_state,
    }
    return result


def _truthy(values: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(values):
        return values.fillna(False).astype(bool)
    numeric = pd.to_numeric(values, errors="coerce")
    return numeric.fillna(0).ne(0)
