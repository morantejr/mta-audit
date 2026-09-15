"""Identity-fragmentation sensitivity audit."""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

from ..attribution import AttributionModel
from ..results import AuditResult
from ..schema import ColumnMapping
from .leakage import _rates, _simulation_audit


def audit_identity_loss(
    events: pd.DataFrame,
    *,
    columns: ColumnMapping | dict[str, str] | None = None,
    rates: Iterable[float] | float = (0.05, 0.10, 0.20),
    models: Iterable[AttributionModel | str] | AttributionModel | str = ("linear",),
    lookback_window: object = "30D",
    conversion_window: object | None = None,
    include_non_converters: bool = True,
    random_state: int | None = 42,
) -> AuditResult:
    """Fragment user journeys and quantify attribution/model sensitivity."""

    normalized = _rates(rates)
    return _simulation_audit(
        "identity_loss",
        events,
        columns=columns,
        scenarios=[(f"{rate:g}", {"identity_loss": rate}, rate) for rate in normalized],
        models=models,
        lookback_window=lookback_window,
        conversion_window=conversion_window,
        include_non_converters=include_non_converters,
        random_state=random_state,
    )
