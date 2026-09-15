"""Path-quality audit compatibility exports."""

from .path_sparsity import PathSparsityMetrics, audit_path_sparsity, calculate_path_sparsity

audit_path_quality = audit_path_sparsity

__all__ = [
    "PathSparsityMetrics",
    "audit_path_quality",
    "audit_path_sparsity",
    "calculate_path_sparsity",
]
