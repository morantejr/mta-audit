"""Compatibility exports for the primary public API."""

from .audit import MTAAudit
from .bootstrap import BootstrapConfig, BootstrapResult
from .results import (
    AuditFinding,
    AuditReport,
    AuditResult,
    AuditScore,
    EvidenceStatus,
    Severity,
)
from .robustness import DecisionRobustnessResult, SensitivityResult
from .schema import ColumnMapping, SchemaError

__all__ = [
    "AuditFinding",
    "AuditReport",
    "AuditResult",
    "AuditScore",
    "BootstrapConfig",
    "BootstrapResult",
    "ColumnMapping",
    "DecisionRobustnessResult",
    "EvidenceStatus",
    "MTAAudit",
    "SchemaError",
    "SensitivityResult",
    "Severity",
]
