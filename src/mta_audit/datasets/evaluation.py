"""Attribution comparison helpers used by Criteo benchmarks and audits."""

from __future__ import annotations

import pandas as pd

from ..attribution.base import ATTRIBUTION_COLUMNS, AttributionResult


def share_table(result: AttributionResult | pd.DataFrame) -> pd.Series:
    """Return a channel-indexed share series that sums to one when credit exists."""

    frame = result.data if isinstance(result, AttributionResult) else result
    missing = set(ATTRIBUTION_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"Attribution table is missing columns: {sorted(missing)}")
    shares = frame.set_index("channel")["share"].astype(float)
    return shares.groupby(level=0).sum()


def attribution_share_drift(
    baseline: AttributionResult | pd.DataFrame,
    other: AttributionResult | pd.DataFrame,
) -> float:
    """Return total-variation distance between two share vectors, in [0, 1]."""

    left = share_table(baseline)
    right = share_table(other)
    aligned = pd.concat([left, right], axis=1, keys=["left", "right"]).fillna(0.0)
    return float(aligned["left"].sub(aligned["right"]).abs().sum() / 2.0)


def rank_correlation(
    baseline: AttributionResult | pd.DataFrame,
    other: AttributionResult | pd.DataFrame,
) -> float:
    """Return Spearman rank correlation of channel shares, 1.0 when both are empty."""

    left = share_table(baseline)
    right = share_table(other)
    aligned = pd.concat([left, right], axis=1, keys=["left", "right"]).fillna(0.0)
    if aligned.empty or aligned["left"].nunique() <= 1 or aligned["right"].nunique() <= 1:
        return 1.0 if aligned["left"].equals(aligned["right"]) else 0.0
    left_rank = aligned["left"].rank(method="average")
    right_rank = aligned["right"].rank(method="average")
    value = left_rank.corr(right_rank, method="pearson")
    return 1.0 if pd.isna(value) else float(value)
