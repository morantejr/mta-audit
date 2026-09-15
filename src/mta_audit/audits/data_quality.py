"""Data-quality audit compatibility exports."""

from ..checks import run_data_quality_checks, run_duplicate_checks

audit_data_quality = run_data_quality_checks
audit_duplicates = run_duplicate_checks

__all__ = [
    "audit_data_quality",
    "audit_duplicates",
    "run_data_quality_checks",
    "run_duplicate_checks",
]
