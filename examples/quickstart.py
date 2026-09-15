"""Runnable mta-audit quickstart."""

import pandas as pd

from mta_audit import BootstrapConfig, MTAAudit

events = pd.DataFrame(
    {
        "user_id": ["a", "a", "a", "b", "b", "c", "c"],
        "timestamp": pd.to_datetime(
            [
                "2026-01-01",
                "2026-01-02",
                "2026-01-03",
                "2026-01-01",
                "2026-01-04",
                "2026-01-02",
                "2026-01-08",
            ]
        ),
        "channel": ["search", "email", "direct", "social", "direct", "email", "search"],
        "conversion": [False, False, True, False, True, False, True],
    }
)

report = MTAAudit(data=events).run(
    attribution_models=["first_touch", "last_touch", "linear", "markov"],
    conversion_windows=[1, 7, 30],
    simulations={"identity_loss": {"rates": [0.05, 0.10], "models": ["linear"]}},
    bootstrap=BootstrapConfig(n_iterations=200, random_state=42),
)

print(report.summary())
print(report.decision_robustness())
print(report.sensitivity_matrix("rank").to_string(index=False))
print(report.to_dataframe().loc[:, ["check", "severity", "check_score"]].to_string(index=False))
