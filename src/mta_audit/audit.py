"""High-level Phase 1 orchestration."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from copy import deepcopy
from datetime import timedelta
from typing import Any

import pandas as pd

from .attribution import (
    AttributionModel,
    FirstTouchAttribution,
    LastTouchAttribution,
    LinearAttribution,
)
from .audits import (
    ContaminationThresholds,
    ConversionWindowThresholds,
    TemporalStabilityThresholds,
    audit_channel_concentration,
    audit_conversion_window,
    audit_delayed_converter_contamination,
    audit_event_loss,
    audit_exposure_comparison,
    audit_identity_loss,
    audit_model_disagreement,
    audit_model_stability,
    audit_path_sparsity,
    audit_touchpoint_loss,
    resolve_attribution_model,
)
from .bootstrap import BootstrapConfig, bootstrap_attribution
from .checks import run_data_quality_checks, run_duplicate_checks
from .journeys import build_journeys
from .results import (
    AuditFinding,
    AuditReport,
    AuditResult,
    AuditScore,
    EvidenceStatus,
    Severity,
)
from .robustness import collect_sensitivity, evaluate_decision_robustness
from .schema import ColumnMapping
from .scoring import ScoringWeights, calculate_reliability_score
from .simulation import deduplicate


class MTAAudit:
    """Validate events, build journeys, run checks, and baseline attribution.

    Parameters are immutable orchestration configuration; the input frame is copied
    during normalization and is never mutated.
    """

    def __init__(
        self,
        data: pd.DataFrame | None = None,
        *,
        columns: ColumnMapping | dict[str, str] | None = None,
        user_col: str | None = None,
        timestamp_col: str | None = None,
        channel_col: str | None = None,
        conversion_col: str | None = None,
        conversion_value_col: str | None = None,
        conversion_timestamp_col: str | None = None,
        campaign_id_col: str | None = None,
        campaign_col: str | None = None,
        impression_col: str | None = None,
        click_col: str | None = None,
        cost_col: str | None = None,
        revenue_col: str | None = None,
        device_col: str | None = None,
        publisher_col: str | None = None,
        creative_id_col: str | None = None,
        conversion_id_col: str | None = None,
        order_id_col: str | None = None,
        lookback_window: str | timedelta | pd.Timedelta = "30D",
        conversion_window: str | timedelta | pd.Timedelta | None = None,
        include_non_converters: bool = True,
        minimum_severity: Severity | str = Severity.INFO,
        attribution_models: Iterable[AttributionModel | str] | None = None,
        scoring_weights: ScoringWeights | Mapping[str, float] | None = None,
    ) -> None:
        direct_columns = {
            "user_id": user_col,
            "timestamp": timestamp_col,
            "channel": channel_col,
            "conversion": conversion_col,
            "conversion_value": conversion_value_col,
            "conversion_timestamp": conversion_timestamp_col,
            "campaign_id": campaign_id_col,
            "campaign": campaign_col,
            "impression": impression_col,
            "click": click_col,
            "cost": cost_col,
            "revenue": revenue_col,
            "device": device_col,
            "publisher": publisher_col,
            "creative_id": creative_id_col,
            "conversion_id": conversion_id_col,
            "order_id": order_id_col,
        }
        configured_directly = {
            key: value for key, value in direct_columns.items() if value is not None
        }
        if columns is not None and configured_directly:
            raise ValueError("Use either columns or individual *_col arguments, not both.")
        self.data = data
        self.columns = ColumnMapping.from_value(configured_directly or columns)
        self.lookback_window = lookback_window
        self.conversion_window = conversion_window
        self.include_non_converters = include_non_converters
        self.minimum_severity = Severity.parse(minimum_severity)
        configured_models = (
            tuple(attribution_models)
            if attribution_models is not None
            else (FirstTouchAttribution(), LastTouchAttribution(), LinearAttribution())
        )
        self.attribution_models = tuple(
            resolve_attribution_model(model) for model in configured_models
        )
        self.scoring_weights = ScoringWeights.from_value(scoring_weights)
        if len({model.name for model in self.attribution_models}) != len(
            self.attribution_models
        ):
            raise ValueError("Attribution model names must be unique.")

    def run(
        self,
        events: pd.DataFrame | None = None,
        *,
        checks: Iterable[str] | str | None = None,
        simulations: Mapping[str, Any] | None = None,
        attribution_models: Iterable[AttributionModel | str] | None = None,
        conversion_windows: Iterable[object] | None = None,
        bootstrap: bool | BootstrapConfig = False,
        n_bootstrap: int = 200,
        random_state: int | None = 42,
    ) -> AuditReport:
        """Run all audit operations and return a structured report.

        Malformed required records raise a schema exception during journey
        construction because producing attribution from them would be misleading.
        """

        source = events if events is not None else self.data
        if source is None:
            raise ValueError("Provide data to MTAAudit(data=...) or run(data).")
        initial_results = (
            run_data_quality_checks(source, self.columns),
            run_duplicate_checks(source, self.columns),
        )
        collection = build_journeys(
            source,
            columns=self.columns,
            lookback_window=self.lookback_window,
            conversion_window=self.conversion_window,
            include_non_converters=self.include_non_converters,
        )
        models = (
            tuple(resolve_attribution_model(model) for model in attribution_models)
            if attribution_models is not None
            else self.attribution_models
        )
        if not models:
            raise ValueError("attribution_models must contain at least one model.")
        if len({model.name for model in models}) != len(models):
            raise ValueError("Attribution model names must be unique.")
        attribution = {model.name: model.attribute(collection) for model in models}
        available = {
            "data_quality": lambda: initial_results[0],
            "duplicates": lambda: initial_results[1],
            "model_disagreement": lambda: audit_model_disagreement(attribution),
            "path_sparsity": lambda: audit_path_sparsity(collection),
            "channel_concentration": lambda: audit_channel_concentration(
                collection, attribution
            ),
            "conversion_window": lambda: self.check_conversion_window(
                windows=(7, 14, 30) if conversion_windows is None else conversion_windows,
                model=models[-1],
                events=source,
            ),
            "converter_contamination": lambda: self.check_delayed_converter_contamination(
                events=source
            ),
            "model_stability": lambda: audit_model_stability(
                collection, model=models[-1]
            ),
            "exposure_comparison": lambda: audit_exposure_comparison(
                source, columns=self.columns
            ),
            "identity_loss": lambda: self.simulate_identity_loss(
                events=source, rates=(0.10,), models=("linear",)
            ),
            "touchpoint_loss": lambda: self.simulate_touchpoint_loss(
                events=source, rates=(0.10,), models=("linear",)
            ),
            "markov_identification": lambda: _markov_identification_result(
                collection, models
            ),
        }
        defaults = tuple(name for name in available if name != "event_loss")
        selected = list(
            _normalize_checks(checks, defaults if checks is None else tuple(available))
        )
        configured_simulations = simulations or {}
        unknown_simulations = sorted(
            set(configured_simulations) - {"identity_loss", "touchpoint_loss", "event_loss"}
        )
        if unknown_simulations:
            raise ValueError(f"Unknown simulations: {unknown_simulations}")
        for name in configured_simulations:
            if name not in selected:
                selected.append(name)
        results_list: list[AuditResult] = []
        for name in selected:
            if name in configured_simulations:
                value = configured_simulations[name]
                options = dict(value) if isinstance(value, Mapping) else {}
                options.setdefault("models", models)
                if value is not True and not isinstance(value, Mapping):
                    if name in {"identity_loss", "touchpoint_loss"}:
                        options["rates"] = value
                    elif value not in (None, False):
                        raise TypeError("event_loss simulation configuration must be a mapping.")
                if value is False or value is None:
                    continue
                method = {
                    "identity_loss": self.simulate_identity_loss,
                    "touchpoint_loss": self.simulate_touchpoint_loss,
                    "event_loss": self.simulate_event_loss,
                }[name]
                results_list.append(method(events=source, **options))
            else:
                results_list.append(available[name]())
        results = tuple(results_list)
        bootstrap_result = None
        if bootstrap:
            bootstrap_config = (
                bootstrap
                if isinstance(bootstrap, BootstrapConfig)
                else BootstrapConfig(
                    n_iterations=n_bootstrap,
                    random_state=random_state,
                )
            )
            bootstrap_result = bootstrap_attribution(collection, models, bootstrap_config)
        additional_scenarios: dict[str, pd.DataFrame] = {}
        if bool(source.duplicated().any()):
            deduplicated = deduplicate(source, columns=self.columns)
            deduplicated_journeys = build_journeys(
                deduplicated,
                columns=self.columns,
                lookback_window=self.lookback_window,
                conversion_window=self.conversion_window,
                include_non_converters=self.include_non_converters,
            )
            preferred = next(
                (model for model in models if model.name == "linear"),
                models[0],
            )
            additional_scenarios["duplicate_handling:deduplicated"] = (
                deepcopy(preferred).attribute(deduplicated_journeys).data
            )
        sensitivity = collect_sensitivity(
            attribution,
            results,
            additional=additional_scenarios,
        )
        decision = evaluate_decision_robustness(sensitivity, bootstrap_result)
        return AuditReport(
            results=results,
            score=calculate_reliability_score(results, self.scoring_weights),
            minimum_severity=self.minimum_severity,
            journey_summary=collection.summary,
            attribution=attribution,
            markov_identified=_markov_is_identified(collection, models),
            channel_grain=_channel_grain(collection),
            bootstrap=bootstrap_result,
            sensitivity=sensitivity,
            decision_analysis=decision,
            sample_info=source.attrs.get("mta_audit_sample"),
        )

    def check_conversion_window(
        self,
        windows: Iterable[object] = (7, 14, 30),
        model: AttributionModel | str = "linear",
        *,
        events: pd.DataFrame | None = None,
        thresholds: ConversionWindowThresholds | None = None,
    ) -> AuditResult:
        """Run conversion-window sensitivity against the configured data."""

        source = self._source(events)
        return audit_conversion_window(
            source,
            columns=self.columns,
            windows=tuple(windows),
            model=model,
            lookback_window=self.lookback_window,
            thresholds=thresholds,
        )

    def check_delayed_converter_contamination(
        self,
        short_window: object = 7,
        reference_window: object = 30,
        *,
        events: pd.DataFrame | None = None,
        thresholds: ContaminationThresholds | None = None,
    ) -> AuditResult:
        """Run delayed-converter contamination against configured data."""

        source = self._source(events)
        return audit_delayed_converter_contamination(
            source,
            columns=self.columns,
            short_window=short_window,
            reference_window=reference_window,
            thresholds=thresholds,
        )

    def check_model_stability(
        self,
        *,
        period: str = "W",
        model: AttributionModel | str = "linear",
        events: pd.DataFrame | None = None,
        thresholds: TemporalStabilityThresholds | None = None,
    ) -> AuditResult:
        """Run temporal model stability for a configurable calendar period."""

        collection = build_journeys(
            self._source(events),
            columns=self.columns,
            lookback_window=self.lookback_window,
            conversion_window=self.conversion_window,
            include_non_converters=self.include_non_converters,
        )
        return audit_model_stability(
            collection,
            model=resolve_attribution_model(model),
            period=period,
            thresholds=thresholds,
        )

    check_temporal_stability = check_model_stability

    def simulate_identity_loss(
        self,
        rates: Iterable[float] | float = (0.05, 0.10, 0.20),
        *,
        models: Iterable[AttributionModel | str] | AttributionModel | str | None = None,
        events: pd.DataFrame | None = None,
        random_state: int | None = 42,
    ) -> AuditResult:
        """Measure sensitivity to deterministic identity fragmentation."""

        return audit_identity_loss(
            self._source(events),
            columns=self.columns,
            rates=rates,
            models=models or self.attribution_models,
            lookback_window=self.lookback_window,
            conversion_window=self.conversion_window,
            include_non_converters=self.include_non_converters,
            random_state=random_state,
        )

    def simulate_touchpoint_loss(
        self,
        rates: Iterable[float] | float = (0.05, 0.10, 0.20),
        *,
        rate: float | None = None,
        models: Iterable[AttributionModel | str] | AttributionModel | str | None = None,
        events: pd.DataFrame | None = None,
        random_state: int | None = 42,
    ) -> AuditResult:
        """Measure sensitivity to generic non-conversion touchpoint loss."""

        if rate is not None:
            rates = rate
        return audit_touchpoint_loss(
            self._source(events),
            columns=self.columns,
            rates=rates,
            models=models or self.attribution_models,
            lookback_window=self.lookback_window,
            conversion_window=self.conversion_window,
            include_non_converters=self.include_non_converters,
            random_state=random_state,
        )

    def simulate_event_loss(
        self,
        impression_loss: float = 0,
        click_loss: float = 0,
        *,
        event_type_col: str | None = None,
        models: Iterable[AttributionModel | str] | AttributionModel | str | None = None,
        events: pd.DataFrame | None = None,
        random_state: int | None = 42,
    ) -> AuditResult:
        """Measure sensitivity to impression- and click-specific event loss."""

        return audit_event_loss(
            self._source(events),
            columns=self.columns,
            impression_loss=impression_loss,
            click_loss=click_loss,
            event_type_col=event_type_col,
            models=models or self.attribution_models,
            lookback_window=self.lookback_window,
            conversion_window=self.conversion_window,
            include_non_converters=self.include_non_converters,
            random_state=random_state,
        )

    def _source(self, events: pd.DataFrame | None) -> pd.DataFrame:
        source = events if events is not None else self.data
        if source is None:
            raise ValueError("Provide data to MTAAudit(data=...) or the check method.")
        return source


def _normalize_checks(
    checks: Iterable[str] | str | None, defaults: tuple[str, ...]
) -> tuple[str, ...]:
    if checks is None:
        return defaults
    requested = (checks,) if isinstance(checks, str) else tuple(checks)
    aliases = {
        "conversion_window": "conversion_window",
        "converter_contamination": "converter_contamination",
        "contamination": "converter_contamination",
        "delayed_converter_contamination": "converter_contamination",
        "model_stability": "model_stability",
        "temporal_stability": "model_stability",
        "converter_exposure": "exposure_comparison",
    }
    normalized = tuple(aliases.get(name, name) for name in requested)
    unknown = sorted(set(normalized) - set(defaults))
    if unknown:
        raise ValueError(f"Unknown checks: {unknown}. Available checks: {sorted(defaults)}")
    return tuple(dict.fromkeys(normalized))


def _markov_is_identified(collection: Any, models: Iterable[Any]) -> bool | None:
    if not any(getattr(model, "name", None) == "markov" for model in models):
        return None
    return collection.summary.non_converter_count > 0


def _channel_grain(collection: Any) -> str:
    unique = collection.summary.unique_channel_count
    return "campaign" if unique > 50 else "channel"


def _markov_identification_result(collection: Any, models: Iterable[Any]) -> AuditResult:
    uses_markov = any(getattr(model, "name", None) == "markov" for model in models)
    non_converters = collection.summary.non_converter_count
    if not uses_markov:
        finding = AuditFinding(
            check="markov_identification",
            severity=Severity.INFO,
            message="Markov was not included in the model set.",
            metric_name="markov_identified",
            value=None,
        )
        return AuditResult(
            "markov_identification",
            (finding,),
            AuditScore(
                0,
                status=EvidenceStatus.NOT_APPLICABLE,
                explanation="Markov was not included in the model set.",
            ),
            EvidenceStatus.NOT_APPLICABLE,
            "Markov was not included in the model set.",
        )
    identified = non_converters > 0
    if identified:
        minimum_journeys = max(10, collection.summary.unique_channel_count * 2)
        if collection.summary.journey_count < minimum_journeys:
            explanation = (
                "Markov has converter and NULL paths but the state space is sparse: "
                f"{collection.summary.journey_count} journeys across "
                f"{collection.summary.unique_channel_count} channels; at least "
                f"{minimum_journeys} journeys are recommended for this diagnostic."
            )
            finding = AuditFinding(
                check="markov_identification",
                severity=Severity.INFO,
                message=explanation,
                metric_name="markov_identified",
                value=True,
                count=non_converters,
                recommendation=(
                    "Collect more complete journeys or group channels before "
                    "interpreting Markov removal weights."
                ),
            )
            return AuditResult(
                "markov_identification",
                (finding,),
                AuditScore(
                    0,
                    status=EvidenceStatus.INSUFFICIENT_EVIDENCE,
                    explanation=explanation,
                ),
                EvidenceStatus.INSUFFICIENT_EVIDENCE,
                explanation,
            )
        finding = AuditFinding(
            check="markov_identification",
            severity=Severity.INFO,
            message=(
                f"Markov removal effects are identified from {non_converters} "
                "observed NULL/non-converter paths."
            ),
            metric_name="markov_identified",
            value=True,
            count=non_converters,
        )
        return AuditResult("markov_identification", (finding,), AuditScore(100))
    finding = AuditFinding(
        check="markov_identification",
        severity=Severity.HIGH,
        message=(
            "No observed NULL/non-converter paths; Markov removal effects are "
            "not identified on this source."
        ),
        metric_name="markov_identified",
        value=False,
        recommendation=(
            "Fit Markov on converter + non-converter journeys before using removal weights."
        ),
    )
    explanation = (
        "No observed NULL/non-converter paths; Markov removal effects are not "
        "identified on this source."
    )
    return AuditResult(
        "markov_identification",
        (finding,),
        AuditScore(
            0,
            status=EvidenceStatus.INSUFFICIENT_EVIDENCE,
            explanation=explanation,
        ),
        EvidenceStatus.INSUFFICIENT_EVIDENCE,
        explanation,
    )
