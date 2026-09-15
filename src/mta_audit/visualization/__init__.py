"""Optional plotting helpers.

Importing this module does not import matplotlib; it is loaded only when a plot
function is called.
"""

from .plots import (
    plot_channel_volatility,
    plot_model_comparison,
    plot_scorecard,
    plot_window_sensitivity,
)

__all__ = [
    "plot_channel_volatility",
    "plot_model_comparison",
    "plot_scorecard",
    "plot_window_sensitivity",
]
