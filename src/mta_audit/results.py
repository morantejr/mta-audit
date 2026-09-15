"""Structured audit result objects."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field, fields, is_dataclass
from datetime import date, datetime, timedelta
from enum import IntEnum, StrEnum
from html import escape
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from .bootstrap import BootstrapResult
    from .robustness import DecisionRobustnessResult, SensitivityResult


class Severity(IntEnum):
    """Finding severity ordered from a passed check to critical."""

    PASS = -1
    INFO = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

    @classmethod
    def parse(cls, value: Severity | str) -> Severity:
        """Normalize a severity value."""

        if isinstance(value, cls):
            return value
        try:
            return cls[value.strip().upper()]
        except (AttributeError, KeyError) as exc:
            choices = ", ".join(member.name.lower() for member in cls)
            raise ValueError(f"severity must be one of: {choices}") from exc


class EvidenceStatus(StrEnum):
    """Whether a diagnostic had enough evidence to be interpreted."""

    EVALUATED = "evaluated"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    NOT_APPLICABLE = "not_applicable"


@dataclass(frozen=True, slots=True)
class AuditFinding:
    """One actionable fact found by an audit check."""

    code: str | None = None
    severity: Severity = Severity.PASS
    message: str = ""
    count: int = 0
    details: Mapping[str, Any] = field(default_factory=dict)
    check: str | None = None
    score: float | None = None
    metric_name: str | None = None
    value: Any = None
    metric_value: Any = None
    recommendation: str | None = None

    def __post_init__(self) -> None:
        identifier = self.check or self.code
        if not isinstance(identifier, str) or not identifier.strip():
            raise ValueError("Finding check cannot be empty.")
        if self.code and self.check and self.code != self.check:
            raise ValueError("Finding code and check must match when both are supplied.")
        if self.count < 0:
            raise ValueError("Finding count cannot be negative.")
        if self.score is not None and not 0 <= self.score <= 100:
            raise ValueError("Finding score must be between 0 and 100.")
        if (
            self.value is not None
            and self.metric_value is not None
            and self.value != self.metric_value
        ):
            raise ValueError("value and metric_value must match when both are supplied.")
        resolved_value = self.metric_value if self.metric_value is not None else self.value
        object.__setattr__(self, "code", identifier)
        object.__setattr__(self, "check", identifier)
        object.__setattr__(self, "severity", Severity.parse(self.severity))
        object.__setattr__(self, "details", MappingProxyType(dict(self.details)))
        object.__setattr__(self, "value", resolved_value)
        object.__setattr__(self, "metric_value", resolved_value)


@dataclass(frozen=True, slots=True)
class AuditScore:
    """Normalized audit-robustness score and its component deductions."""

    value: float
    deductions: Mapping[str, float] = field(default_factory=dict)
    components: Mapping[str, float] = field(default_factory=dict)
    weights: Mapping[str, float] = field(default_factory=dict)
    status: EvidenceStatus = EvidenceStatus.EVALUATED
    explanation: str | None = None

    def __post_init__(self) -> None:
        if not 0 <= self.value <= 100:
            raise ValueError("Audit score must be between 0 and 100.")
        object.__setattr__(self, "deductions", MappingProxyType(dict(self.deductions)))
        object.__setattr__(self, "components", MappingProxyType(dict(self.components)))
        object.__setattr__(self, "weights", MappingProxyType(dict(self.weights)))
        object.__setattr__(self, "status", EvidenceStatus(self.status))

    @property
    def grade(self) -> str:
        """Legacy letter-grade compatibility alias.

        Prefer :attr:`label`; an audit score is not percent correctness.
        """

        if self.value >= 90:
            return "A"
        if self.value >= 80:
            return "B"
        if self.value >= 70:
            return "C"
        if self.value >= 60:
            return "D"
        return "F"

    @property
    def label(self) -> str:
        """Return the centralized qualitative audit-score interpretation."""

        if self.status is EvidenceStatus.INSUFFICIENT_EVIDENCE:
            return "Insufficient Evidence"
        if self.status is EvidenceStatus.NOT_APPLICABLE:
            return "Not Applicable"
        from .scoring import score_label

        return score_label(self.value)


@dataclass(frozen=True, slots=True)
class AuditResult:
    """Result from one named audit area."""

    name: str
    findings: tuple[AuditFinding, ...]
    score: AuditScore
    status: EvidenceStatus = EvidenceStatus.EVALUATED
    explanation: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "status", EvidenceStatus(self.status))

    def visible_findings(self, minimum: Severity | str = Severity.INFO) -> tuple[AuditFinding, ...]:
        """Return findings at or above the requested threshold."""

        threshold = Severity.parse(minimum)
        return tuple(finding for finding in self.findings if finding.severity >= threshold)

    @property
    def passed(self) -> bool:
        """Whether the audit contains no high or critical findings."""

        return not any(finding.severity >= Severity.HIGH for finding in self.findings)


@dataclass(frozen=True, slots=True)
class AuditReport:
    """Complete attribution reliability report."""

    results: tuple[AuditResult, ...]
    score: AuditScore
    minimum_severity: Severity = Severity.INFO
    journey_summary: Any | None = None
    attribution: Mapping[str, Any] = field(default_factory=dict)
    markov_identified: bool | None = None
    channel_grain: str = "channel"
    bootstrap: BootstrapResult | None = None
    sensitivity: SensitivityResult | None = None
    decision_analysis: DecisionRobustnessResult | None = None
    sample_info: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "minimum_severity", Severity.parse(self.minimum_severity))
        object.__setattr__(self, "attribution", MappingProxyType(dict(self.attribution)))
        if self.sample_info is not None:
            object.__setattr__(self, "sample_info", MappingProxyType(dict(self.sample_info)))

    @property
    def findings(self) -> tuple[AuditFinding, ...]:
        """Return report-visible findings in result order."""

        return tuple(
            finding
            for result in self.results
            for finding in result.visible_findings(self.minimum_severity)
        )

    def summary(self) -> str:
        """Return a compact, readable scorecard with actionable findings."""

        counts = {
            severity: sum(finding.severity is severity for finding in self.findings)
            for severity in Severity
            if severity is not Severity.PASS
        }
        markov = (
            "identified"
            if self.markov_identified
            else "not identified"
            if self.markov_identified is False
            else "not evaluated"
        )
        overall = (
            f"{self.score.value:.1f}/100 — {self.score.label}"
            if self.score.status is EvidenceStatus.EVALUATED
            else f"not evaluated — {self.score.explanation}"
        )
        lines = [
            "MTA Audit Reliability Report",
            f"Overall audit score: {overall}",
            "Component scores:",
            *[
                f"- {name.replace('_', ' ').title()}: {value:.1f}"
                for name, value in self.score.components.items()
            ],
            f"Checks run: {len(self.results)} | Visible findings: {len(self.findings)}",
            "Severity counts: "
            + ", ".join(f"{level.name.lower()}={counts[level]}" for level in counts),
            f"Markov removal effects: {markov}",
            f"Channel grain: {self.channel_grain}",
        ]
        insufficient = [
            result for result in self.results
            if result.status is EvidenceStatus.INSUFFICIENT_EVIDENCE
        ]
        if insufficient:
            lines.append("Insufficient evidence:")
            lines.extend(
                f"- {result.name}: {result.explanation or 'not enough evidence'}"
                for result in insufficient
            )
        if self.bootstrap is not None and not self.bootstrap.evaluated:
            lines.append(
                "Bootstrap uncertainty: insufficient evidence — "
                f"{self.bootstrap.explanation}"
            )
        if self.decision_analysis is not None:
            lines.append("Decision robustness:")
            lines.append(self.decision_analysis.interpretation)
        recommendations = list(
            dict.fromkeys(
                finding.recommendation
                for finding in self.findings
                if finding.recommendation
            )
        )
        if recommendations:
            lines.append("Recommendations:")
            lines.extend(f"- {item}" for item in recommendations)
        else:
            lines.append("Recommendations: No visible findings require action.")
        return "\n".join(lines)

    def to_dataframe(self) -> pd.DataFrame:
        """Return one serialization-safe row per finding, including passed findings."""

        rows: list[dict[str, Any]] = []
        for result in self.results:
            if not result.findings:
                rows.append(
                    {
                        "result": result.name,
                        "check": result.name,
                        "severity": "pass",
                        "message": "",
                        "count": 0,
                        "check_score": result.score.value,
                        "report_score": self.score.value,
                        "result_status": result.status.value,
                        "result_explanation": result.explanation,
                        "metric_name": None,
                        "value": None,
                        "metric_value": None,
                        "recommendation": None,
                    }
                )
            for finding in result.findings:
                rows.append(
                    {
                        "result": result.name,
                        "check": finding.check,
                        "severity": finding.severity.name.lower(),
                        "message": finding.message,
                        "count": finding.count,
                        "finding_score": finding.score,
                        "check_score": result.score.value,
                        "report_score": self.score.value,
                        "result_status": result.status.value,
                        "result_explanation": result.explanation,
                        "metric_name": finding.metric_name,
                        "value": _jsonable(finding.value),
                        "metric_value": _jsonable(finding.metric_value),
                        "details": _jsonable(finding.details),
                        "recommendation": finding.recommendation,
                    }
                )
        return pd.DataFrame(rows)

    def to_json(
        self,
        *,
        indent: int | None = 2,
        include_bootstrap_draws: bool = False,
    ) -> str:
        """Serialize report metrics, optionally including replicate-level draws."""

        payload = {
            "score": _jsonable(self.score),
            "minimum_severity": self.minimum_severity.name.lower(),
            "journey_summary": _jsonable(self.journey_summary),
            "results": _jsonable(self.results),
            "attribution": _jsonable(self.attribution),
            "markov_identified": self.markov_identified,
            "channel_grain": self.channel_grain,
            "bootstrap": _bootstrap_json(
                self.bootstrap,
                include_draws=include_bootstrap_draws,
            ),
            "sensitivity": _jsonable(self.sensitivity),
            "decision_robustness": _jsonable(self.decision_analysis),
            "sample_info": _jsonable(self.sample_info),
        }
        return json.dumps(payload, indent=indent, allow_nan=False)

    def sensitivity_matrix(self, metric: str = "rank") -> pd.DataFrame:
        """Return channel rank/share under the assumptions evaluated by ``run``."""

        if self.sensitivity is None:
            return pd.DataFrame(columns=["channel"])
        return self.sensitivity.matrix(metric)

    def decision_robustness(self) -> DecisionRobustnessResult:
        """Return stability of channel decisions across tested assumptions."""

        if self.decision_analysis is None:
            from .robustness import DecisionRobustnessResult, empty_decision_frame

            return DecisionRobustnessResult(
                "insufficient_evidence",
                "Decision scenarios were not collected.",
                empty_decision_frame(),
                "Decision robustness was not evaluated: no scenarios were collected.",
            )
        return self.decision_analysis

    def to_markdown(self) -> str:
        """Render a standalone, shareable observational-stability report."""

        caveat = (
            "These diagnostics assess the stability and robustness of observational "
            "attribution results. They do not estimate causal incrementality."
        )
        sections = [
            "# MTA Audit Reliability Report",
            caveat,
            (
                "## Overall audit score\n\n"
                + (
                    f"**{self.score.value:.1f}/100 — {self.score.label}**"
                    if self.score.status is EvidenceStatus.EVALUATED
                    else f"**Not evaluated — {self.score.explanation}**"
                )
            ),
            "## Component scorecard\n\n"
            + _markdown_table(
                pd.DataFrame(
                    [
                        {"component": key, "score": round(value, 1)}
                        for key, value in self.score.components.items()
                    ]
                )
            ),
            "## Major warnings\n\n"
            + (
                "\n".join(
                    f"- **{finding.severity.name.title()} — {finding.check}:** "
                    f"{finding.message}"
                    for finding in self.findings
                    if finding.severity >= Severity.MEDIUM
                )
                or "No medium-or-higher findings."
            ),
        ]
        if self.attribution:
            attribution = pd.concat(
                [
                    output.data.assign(model=name)
                    for name, output in self.attribution.items()
                ],
                ignore_index=True,
            )
            sections.append("## Attribution outputs\n\n" + _markdown_table(attribution))
        if self.bootstrap is not None:
            sections.append(
                "## Bootstrap uncertainty\n\n"
                + (
                    _markdown_table(self.bootstrap.statistics)
                    if self.bootstrap.evaluated
                    else f"Insufficient evidence: {self.bootstrap.explanation}"
                )
            )
        if self.decision_analysis is not None:
            sections.append(
                "## Decision robustness\n\n"
                + self.decision_analysis.interpretation
                + "\n\n"
                + _markdown_table(self.decision_analysis.statistics)
            )
        matrix = self.sensitivity_matrix("rank")
        if not matrix.empty:
            sections.append("## Assumption sensitivity (rank)\n\n" + _markdown_table(matrix))
        sections.append("## Data-quality findings\n\n" + _markdown_table(self.to_dataframe()))
        sections.append(
            "## Methodology caveats\n\n"
            "- Bootstrap intervals describe sampling variability in the observed "
            "journeys, not causal uncertainty.\n"
            "- Scenario frequencies describe the tested assumptions only.\n"
            "- Attribution share is allocated observational credit, not lift."
        )
        sections.append(
            "## What these results do not mean\n\n"
            "The audit score is not the probability that attribution is correct. "
            "No result estimates incremental conversions or causal return."
        )
        return "\n\n".join(sections)

    def to_html(self) -> str:
        """Render a standalone HTML report without a server or dashboard."""

        caveat = (
            "These diagnostics assess the stability and robustness of observational "
            "attribution results. They do not estimate causal incrementality."
        )
        components = pd.DataFrame(
            [
                {"component": key, "score": round(value, 1)}
                for key, value in self.score.components.items()
            ]
        )
        warnings = [
            finding
            for finding in self.findings
            if finding.severity >= Severity.MEDIUM
        ]
        blocks: list[str] = [
            "<!doctype html><html><head><meta charset='utf-8'>",
            "<title>MTA Audit Reliability Report</title>",
            "<style>body{font:15px system-ui;max-width:1100px;margin:40px auto;"
            "padding:0 20px;line-height:1.45}table{border-collapse:collapse;"
            "width:100%;overflow:auto}th,td{border:1px solid #ddd;padding:6px;"
            "text-align:left}h1,h2{margin-top:1.4em}.caveat{padding:12px;"
            "background:#fff4cc;border-left:4px solid #b8860b}</style></head><body>",
            "<h1>MTA Audit Reliability Report</h1>",
            f"<p class='caveat'>{escape(caveat)}</p>",
            "<h2>Overall audit score</h2><p><strong>"
            + (
                f"{self.score.value:.1f}/100 — {escape(self.score.label)}"
                if self.score.status is EvidenceStatus.EVALUATED
                else f"Not evaluated — {escape(str(self.score.explanation))}"
            )
            + "</strong></p>",
            "<h2>Component scorecard</h2>",
            components.to_html(index=False, border=0),
            "<h2>Major warnings</h2>",
        ]
        blocks.extend(
            (
                f"<p><strong>{escape(finding.severity.name.title())} — "
                f"{escape(str(finding.check))}:</strong> "
                f"{escape(finding.message)}</p>"
            )
            for finding in warnings
        )
        if not warnings:
            blocks.append("<p>No medium-or-higher findings.</p>")
        if self.attribution:
            attribution = pd.concat(
                [
                    output.data.assign(model=name)
                    for name, output in self.attribution.items()
                ],
                ignore_index=True,
            )
            blocks.extend(
                [
                    "<h2>Attribution outputs</h2>",
                    attribution.to_html(index=False, border=0),
                ]
            )
        if self.bootstrap is not None:
            blocks.append("<h2>Bootstrap uncertainty</h2>")
            blocks.append(
                self.bootstrap.statistics.to_html(index=False, border=0)
                if self.bootstrap.evaluated
                else f"<p>Insufficient evidence: {escape(str(self.bootstrap.explanation))}</p>"
            )
        if self.decision_analysis is not None:
            blocks.extend(
                [
                    "<h2>Decision robustness</h2>",
                    f"<p>{escape(self.decision_analysis.interpretation)}</p>",
                    self.decision_analysis.statistics.to_html(index=False, border=0),
                ]
            )
        matrix = self.sensitivity_matrix("rank")
        if not matrix.empty:
            blocks.extend(
                [
                    "<h2>Assumption sensitivity (rank)</h2>",
                    matrix.to_html(index=False, border=0),
                ]
            )
        blocks.extend(
            [
                "<h2>Data-quality findings</h2>",
                self.to_dataframe().to_html(index=False, border=0),
                "<h2>Methodology caveats</h2>",
                (
                    "<ul><li>Bootstrap intervals describe observed-journey sampling "
                    "variability, not causal uncertainty.</li><li>Scenario frequencies "
                    "describe tested assumptions only.</li><li>Attribution share is "
                    "observational credit, not lift.</li></ul>"
                ),
                "<h2>What these results do not mean</h2>",
                (
                    "<p>The audit score is not the probability that attribution is "
                    "correct. No result estimates incremental conversions or causal "
                    "return.</p>"
                ),
                "</body></html>",
            ]
        )
        return "".join(blocks)

    def plot_scorecard(self, **kwargs: Any) -> Any:
        """Plot component reliability scores (requires the ``viz`` extra)."""

        from .visualization.plots import plot_scorecard

        return plot_scorecard(self, **kwargs)

    def plot_model_comparison(self, **kwargs: Any) -> Any:
        """Plot attribution shares by model (requires the ``viz`` extra)."""

        from .visualization.plots import plot_model_comparison

        return plot_model_comparison(self, **kwargs)

    def plot_window_sensitivity(self, **kwargs: Any) -> Any:
        """Plot conversion-window sensitivity (requires the ``viz`` extra)."""

        from .visualization.plots import plot_window_sensitivity

        return plot_window_sensitivity(self, **kwargs)

    def plot_channel_volatility(self, **kwargs: Any) -> Any:
        """Plot channel volatility across windows or periods."""

        from .visualization.plots import plot_channel_volatility

        return plot_channel_volatility(self, **kwargs)


def _jsonable(value: Any) -> Any:
    """Recursively convert pandas/numpy/dataclass values to strict JSON values."""

    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, IntEnum):
        return value.name.lower()
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, pd.DataFrame):
        return [_jsonable(row) for row in value.to_dict(orient="records")]
    if isinstance(value, pd.Series):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, pd.Timestamp | datetime | date):
        return value.isoformat()
    if isinstance(value, pd.Timedelta | timedelta):
        return str(value)
    if is_dataclass(value) and not isinstance(value, type):
        return {
            item.name: _jsonable(getattr(value, item.name))
            for item in fields(value)
        }
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, set)):
        return [_jsonable(item) for item in value]
    return str(value)


def _bootstrap_json(value: Any, *, include_draws: bool) -> Any:
    if value is None:
        return None
    payload = {
        "status": _jsonable(value.status),
        "explanation": value.explanation,
        "config": _jsonable(value.config),
        "statistics": _jsonable(value.statistics),
    }
    if include_draws:
        payload["draws"] = _jsonable(value.draws)
    return payload


def _markdown_table(frame: pd.DataFrame, *, max_rows: int = 50) -> str:
    """Render a compact table without requiring the optional tabulate package."""

    if frame.empty:
        return "No evaluated rows."
    display = frame.head(max_rows).copy()
    columns = [str(column) for column in display.columns]

    def cell(value: Any) -> str:
        if value is None or (not isinstance(value, (list, dict)) and pd.isna(value)):
            return ""
        if isinstance(value, float):
            return f"{value:.4g}"
        return str(value).replace("|", "\\|").replace("\n", " ")

    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join("---" for _ in columns) + " |"
    rows = [
        "| " + " | ".join(cell(value) for value in row) + " |"
        for row in display.itertuples(index=False, name=None)
    ]
    if len(frame) > max_rows:
        rows.append(f"\n_Showing {max_rows} of {len(frame)} rows._")
    return "\n".join([header, divider, *rows])
