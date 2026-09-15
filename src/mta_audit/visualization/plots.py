"""Matplotlib visualizations for :class:`mta_audit.AuditReport`."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pandas as pd

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from ..results import AuditReport, AuditResult


def _axes(ax: Axes | None = None, *, figsize: tuple[float, float] = (8, 4)) -> Axes:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "Plotting requires matplotlib. Install it with `pip install 'mta-audit[viz]'`."
        ) from exc
    if ax is not None:
        return ax
    _, created = plt.subplots(figsize=figsize)
    return created


def plot_scorecard(
    report: AuditReport, *, ax: Axes | None = None, title: str = "MTA reliability scorecard"
) -> Axes:
    """Plot completed scoring components and their normalized weights."""

    axis = _axes(ax)
    components = report.score.components
    if not components:
        raise ValueError("This report has no scored components.")
    names = [name.replace("_", " ").title() for name in components]
    values = list(components.values())
    colors = [
        "#2e7d32" if value >= 80 else "#f9a825" if value >= 60 else "#c62828"
        for value in values
    ]
    axis.barh(names, values, color=colors)
    axis.set(xlim=(0, 100), xlabel="Reliability score", title=title)
    axis.axvline(80, color="#555555", linewidth=1, linestyle="--")
    axis.invert_yaxis()
    return axis


def plot_model_comparison(
    report: AuditReport, *, ax: Axes | None = None, title: str = "Attribution by model"
) -> Axes:
    """Plot channel share for every attribution model in a report."""

    axis = _axes(ax)
    frames: dict[str, pd.Series] = {}
    for model, result in report.attribution.items():
        frame = result.data if hasattr(result, "data") else result
        if isinstance(frame, pd.DataFrame) and {"channel", "share"}.issubset(frame):
            frames[model] = frame.set_index("channel")["share"]
    if not frames:
        raise ValueError("This report has no attribution results to compare.")
    pd.DataFrame(frames).fillna(0).plot.bar(ax=axis)
    axis.set(ylabel="Attribution share", xlabel="Channel", title=title)
    axis.legend(title="Model")
    return axis


def plot_window_sensitivity(
    report: AuditReport, *, ax: Axes | None = None, title: str = "Window sensitivity"
) -> Axes:
    """Plot channel attribution shares across tested conversion windows."""

    result = _result(report, "conversion_window")
    metrics = _metrics(result)
    attribution = getattr(metrics, "attribution", None)
    if not attribution:
        raise ValueError("The conversion-window result has no attribution series.")
    axis = _axes(ax)
    frame = pd.DataFrame(
        {
            window: values.set_index("channel")["share"]
            for window, values in attribution.items()
        }
    ).fillna(0)
    frame.plot(ax=axis, marker="o")
    axis.set(ylabel="Attribution share", xlabel="Channel", title=title)
    axis.legend(title="Window")
    return axis


def plot_channel_volatility(
    report: AuditReport, *, ax: Axes | None = None, title: str = "Channel volatility"
) -> Axes:
    """Plot per-channel share range across windows or calendar periods."""

    axis = _axes(ax)
    try:
        result = _result(report, "conversion_window")
        comparison = _metrics(result).channel_comparison
        values = comparison.set_index("channel")["share_volatility"].sort_values()
    except (ValueError, AttributeError, KeyError):
        result = _result(report, "model_stability")
        attribution = getattr(_metrics(result), "attribution", {})
        shares = pd.DataFrame(
            {
                period: frame.set_index("channel")["share"]
                for period, frame in attribution.items()
            }
        ).fillna(0)
        values = (shares.max(axis=1) - shares.min(axis=1)).sort_values()
    values.plot.barh(ax=axis, color="#1565c0")
    axis.set(xlabel="Maximum share range", ylabel="Channel", title=title)
    return axis


def _result(report: AuditReport, name: str) -> AuditResult:
    try:
        return next(result for result in report.results if result.name == name)
    except StopIteration as exc:
        raise ValueError(f"Run the {name!r} check before requesting this plot.") from exc


def _metrics(result: AuditResult) -> Any:
    for finding in result.findings:
        if "metrics" in finding.details:
            return finding.details["metrics"]
    raise ValueError(f"Audit result {result.name!r} has no plottable metrics.")
