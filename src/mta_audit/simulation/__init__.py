"""Public simulation API."""

from .corruption import (
    CorruptedFrame,
    CorruptionMetadata,
    corrupt,
    deduplicate,
    detect_duplicates,
)

__all__ = [
    "CorruptedFrame",
    "CorruptionMetadata",
    "corrupt",
    "deduplicate",
    "detect_duplicates",
]
