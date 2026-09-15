"""Statistical audits for attribution inputs and outputs."""

from .channel_concentration import (
    ChannelConcentrationMetrics,
    ConcentrationStats,
    audit_channel_concentration,
    calculate_channel_concentration,
)
from .contamination import (
    ContaminationMetrics,
    ContaminationThresholds,
    audit_delayed_converter_contamination,
)
from .conversion_window import (
    ConversionWindowMetrics,
    ConversionWindowThresholds,
    audit_conversion_window,
    resolve_attribution_model,
)
from .exposure import (
    ExposureComparisonMetrics,
    ExposureGroupStats,
    ExposureThresholds,
    audit_exposure_comparison,
    calculate_exposure_comparison,
)
from .identity import audit_identity_loss
from .leakage import (
    LeakageSimulationMetrics,
    SimulationScenario,
    audit_event_loss,
    audit_touchpoint_loss,
)
from .model_disagreement import (
    ModelDisagreementMetrics,
    audit_model_disagreement,
    calculate_model_disagreement,
)
from .model_stability import (
    TemporalStabilityMetrics,
    TemporalStabilityThresholds,
    audit_model_stability,
    audit_temporal_stability,
)
from .path_sparsity import (
    PathSparsityMetrics,
    audit_path_sparsity,
    calculate_path_sparsity,
)

__all__ = [
    "ChannelConcentrationMetrics",
    "ConcentrationStats",
    "ContaminationMetrics",
    "ContaminationThresholds",
    "ConversionWindowMetrics",
    "ConversionWindowThresholds",
    "ExposureComparisonMetrics",
    "ExposureGroupStats",
    "ExposureThresholds",
    "LeakageSimulationMetrics",
    "ModelDisagreementMetrics",
    "PathSparsityMetrics",
    "SimulationScenario",
    "TemporalStabilityMetrics",
    "TemporalStabilityThresholds",
    "audit_channel_concentration",
    "audit_conversion_window",
    "audit_delayed_converter_contamination",
    "audit_event_loss",
    "audit_exposure_comparison",
    "audit_identity_loss",
    "audit_model_disagreement",
    "audit_model_stability",
    "audit_path_sparsity",
    "audit_temporal_stability",
    "audit_touchpoint_loss",
    "calculate_channel_concentration",
    "calculate_exposure_comparison",
    "calculate_model_disagreement",
    "calculate_path_sparsity",
    "resolve_attribution_model",
]
