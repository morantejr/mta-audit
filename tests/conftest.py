"""Pytest configuration."""

from __future__ import annotations

import pytest


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Keep Criteo integration tests opt-in via ``pytest -m criteo``."""

    markexpr = config.option.markexpr or ""
    if "criteo" in markexpr and "not criteo" not in markexpr:
        return
    items[:] = [item for item in items if "criteo" not in item.keywords]
