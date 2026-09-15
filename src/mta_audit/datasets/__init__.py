"""Dataset adapters kept outside the general attribution engine."""

from .criteo import (
    CRITEO_COLUMNS,
    CRITEO_DATASET_URL,
    CriteoAttributionAdapter,
    DatasetNotAvailableError,
    load_criteo_attribution,
    transform_criteo_events,
)
from .evaluation import attribution_share_drift, rank_correlation, share_table
from .external import (
    IGNATIO_TOUCHPOINTS_URL,
    TRIANGULATION_MTA_URL,
    load_ignazio_attribution,
    load_triangulation_mta,
    simulate_criteo_label_events,
    simulate_jd_mta_events,
    simulate_synthesizer_events,
    transform_ignazio_touchpoints,
    transform_synthesizer_touchpoints,
    transform_triangulation_mta,
)
from .scenarios import SyntheticStressScenario, make_stress_scenario
from .synthetic import (
    SYNTHETIC_JOURNEYS_URL,
    SYNTHETIC_TOUCHPOINTS_URL,
    load_synthetic_attribution,
    transform_synthetic_attribution,
)

__all__ = [
    "CRITEO_COLUMNS",
    "CRITEO_DATASET_URL",
    "IGNATIO_TOUCHPOINTS_URL",
    "SYNTHETIC_JOURNEYS_URL",
    "SYNTHETIC_TOUCHPOINTS_URL",
    "TRIANGULATION_MTA_URL",
    "CriteoAttributionAdapter",
    "DatasetNotAvailableError",
    "SyntheticStressScenario",
    "attribution_share_drift",
    "load_criteo_attribution",
    "load_ignazio_attribution",
    "load_synthetic_attribution",
    "load_triangulation_mta",
    "make_stress_scenario",
    "rank_correlation",
    "share_table",
    "simulate_criteo_label_events",
    "simulate_jd_mta_events",
    "simulate_synthesizer_events",
    "transform_criteo_events",
    "transform_ignazio_touchpoints",
    "transform_synthesizer_touchpoints",
    "transform_synthetic_attribution",
    "transform_triangulation_mta",
]
