"""Deterministic baseline attribution models."""

from __future__ import annotations

from ..journeys import Journey
from .base import AttributionModel


class FirstTouchAttribution(AttributionModel):
    """Assign all credit to the first touch."""

    name = "first_touch"

    def _weights(self, journey: Journey) -> list[tuple[str, float]]:
        return [(journey.first_channel, 1.0)]


class LastTouchAttribution(AttributionModel):
    """Assign all credit to the last touch."""

    name = "last_touch"

    def _weights(self, journey: Journey) -> list[tuple[str, float]]:
        return [(journey.last_channel, 1.0)]


class LinearAttribution(AttributionModel):
    """Assign equal credit to every touch, including repeated channels."""

    name = "linear"

    def _weights(self, journey: Journey) -> list[tuple[str, float]]:
        return [(channel, 1.0) for channel in journey.channels]
