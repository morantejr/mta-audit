#!/usr/bin/env python3
"""Reproducible Criteo reliability benchmark for mta-audit.

This script downloads (optionally) the public Criteo Attribution Modeling
dataset, maps it through the Criteo adapter, and measures whether known
tracking corruptions change attribution and the reliability score.

Sampling is explicit. The default is a deterministic user sample because
materializing 6M+ journey objects from 16.5M impressions is memory-heavy.
Pass ``--sample-users 0`` to load every user in the local file.
"""

from __future__ import annotations

import argparse
import json
import resource
import sys
import time
import tracemalloc
from pathlib import Path
from typing import Any

import pandas as pd

from mta_audit import MTAAudit, build_journeys
from mta_audit.attribution import (
    FirstTouchAttribution,
    LastTouchAttribution,
    LinearAttribution,
    MarkovAttribution,
)
from mta_audit.datasets import (
    DatasetNotAvailableError,
    attribution_share_drift,
    load_criteo_attribution,
    rank_correlation,
)
from mta_audit.datasets.criteo import describe_sampling
from mta_audit.simulation import corrupt

CHECKS = (
    "data_quality",
    "duplicates",
    "model_disagreement",
    "path_sparsity",
    "channel_concentration",
    "converter_contamination",
    "exposure_comparison",
)
DRIFT_CHECKS = (
    "data_quality",
    "duplicates",
    "model_disagreement",
    "path_sparsity",
    "channel_concentration",
    "converter_contamination",
)
MODELS = ("first_touch", "last_touch", "linear", "markov")
IDENTITY_RATES = (0.0, 0.05, 0.10, 0.20, 0.30)
TOUCHPOINT_RATES = (0.0, 0.05, 0.10, 0.20, 0.30)
DUPLICATE_RATES = (0.01, 0.05, 0.10)
WINDOWS = (1, 3, 7, 14, 30)


def _rss_mb() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return usage / (1024 * 1024)
    return usage / 1024


def _timed(label: str, times: dict[str, float], fn):  # type: ignore[no-untyped-def]
    started = time.perf_counter()
    result = fn()
    times[label] = time.perf_counter() - started
    return result


def _run_audit(events: pd.DataFrame, checks: tuple[str, ...] = CHECKS) -> Any:
    return MTAAudit(data=events, lookback_window="30D").run(
        attribution_models=list(MODELS),
        conversion_windows=WINDOWS,
        checks=list(checks),
    )


def _linear_result(report) -> Any:
    return report.attribution["linear"]


def _row(
    scenario: str,
    report,
    baseline_report=None,
) -> dict[str, Any]:
    linear = _linear_result(report)
    drift = 0.0
    corr = 1.0
    if baseline_report is not None:
        drift = attribution_share_drift(_linear_result(baseline_report), linear)
        corr = rank_correlation(_linear_result(baseline_report), linear)
    return {
        "scenario": scenario,
        "reliability": round(report.score.value, 2),
        "attribution_drift": round(drift, 4),
        "rank_corr": round(corr, 4),
        "journeys": getattr(report.journey_summary, "journey_count", None),
        "unique_paths": getattr(report.journey_summary, "unique_path_count", None),
        "avg_path_length": getattr(report.journey_summary, "average_path_length", None),
        "model_agreement": report.score.components.get("model_agreement"),
        "data_quality": report.score.components.get("data_quality"),
        "path_quality": report.score.components.get("path_quality"),
    }


def _markdown_table(rows: list[dict[str, Any]]) -> str:
    header = (
        "Scenario                  Rel   Drift  RankCorr  Journeys  UniquePaths\n"
        "---------------------------------------------------------------------"
    )
    lines = [header]
    for row in rows:
        lines.append(
            f"{row['scenario']:<24} {row['reliability']:>6.2f} "
            f"{row['attribution_drift']:>7.4f} {row['rank_corr']:>9.4f} "
            f"{int(row['journeys'] or 0):>9} {int(row['unique_paths'] or 0):>12}"
        )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", default=None, help="Local Criteo TSV or TSV.GZ path")
    parser.add_argument("--download", action="store_true", help="Download the public file if missing")
    parser.add_argument("--sample-users", type=int, default=8000)
    parser.add_argument(
        "--output-dir",
        default="benchmarks/results",
        help="Directory for JSON, markdown, and channel tables",
    )
    args = parser.parse_args()
    sample_users = None if args.sample_users == 0 else args.sample_users
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    times: dict[str, float] = {}
    tracemalloc.start()
    try:
        events = _timed(
            "load",
            times,
            lambda: load_criteo_attribution(
                path=args.path,
                download=args.download,
                sample_users=sample_users,
                random_state=42,
            ),
        )
    except DatasetNotAvailableError as exc:
        print(exc)
        return 2

    sampling_note = describe_sampling(
        sample_users,
        int(events["user_id"].nunique()),
        "16.5M impressions / ~700 campaigns in the full public file",
    )
    journeys = _timed(
        "journeys",
        times,
        lambda: build_journeys(events, lookback_window="30D"),
    )
    def _attribute_all() -> dict[str, Any]:
        return {
            "first_touch": FirstTouchAttribution().fit(journeys).attribute(),
            "last_touch": LastTouchAttribution().fit(journeys).attribute(),
            "linear": LinearAttribution().fit(journeys).attribute(),
            "markov": MarkovAttribution().fit(journeys).attribute(),
        }

    model_results = _timed("attribution", times, _attribute_all)
    baseline = _timed("baseline_audit", times, lambda: _run_audit(events))
    current, peak = tracemalloc.get_traced_memory()
    memory = {
        "tracemalloc_current_mb": current / (1024 * 1024),
        "tracemalloc_peak_mb": peak / (1024 * 1024),
        "ru_maxrss_mb": _rss_mb(),
    }

    identity_rows = []
    identity_reports = {}
    for rate in IDENTITY_RATES:
        if rate == 0:
            report = MTAAudit(data=events, lookback_window="30D").run(
                attribution_models=list(MODELS),
                checks=list(DRIFT_CHECKS),
            )
            scenario = "Baseline"
        else:
            corrupted = corrupt(events, identity_loss=rate, random_state=42).to_dataframe()
            report = MTAAudit(data=corrupted, lookback_window="30D").run(
                attribution_models=list(MODELS),
                checks=list(DRIFT_CHECKS),
            )
            scenario = f"{int(rate * 100)}% identity loss"
        identity_reports[scenario] = report
        identity_rows.append(_row(scenario, report, None if rate == 0 else identity_reports["Baseline"]))

    touch_rows = []
    touch_baseline = identity_reports["Baseline"]
    for rate in TOUCHPOINT_RATES:
        if rate == 0:
            touch_rows.append(_row("Baseline", touch_baseline))
            continue
        corrupted = corrupt(events, touchpoint_loss=rate, random_state=42).to_dataframe()
        report = MTAAudit(data=corrupted, lookback_window="30D").run(
            attribution_models=list(MODELS),
            checks=list(DRIFT_CHECKS),
        )
        touch_rows.append(_row(f"{int(rate * 100)}% touchpoint loss", report, touch_baseline))

    duplicate_rows = []
    for rate in DUPLICATE_RATES:
        corrupted = corrupt(events, duplicate_rate=rate, random_state=42)
        report = MTAAudit(data=corrupted.to_dataframe(), lookback_window="30D").run(
            attribution_models=["linear"],
            checks=["data_quality", "duplicates"],
        )
        detected = any(
            finding.code in {"exact_duplicates", "event_identity_duplicates"}
            for result in report.results
            for finding in result.findings
        )
        duplicate_rows.append(
            {
                "scenario": f"{int(rate * 100)}% duplicate events",
                "reliability": round(report.score.value, 2),
                "duplicates_detected": detected,
                "duplicate_rows_injected": corrupted.metadata.row_counts.get("duplicate_rows_added"),
            }
        )

    window = MTAAudit(data=events, lookback_window="30D").check_conversion_window(
        windows=WINDOWS, model="linear"
    )
    contamination = MTAAudit(data=events, lookback_window="30D").check_delayed_converter_contamination(
        short_window=7, reference_window=30
    )

    payload = {
        "sampling": sampling_note,
        "n_events": int(len(events)),
        "n_users": int(events["user_id"].nunique()),
        "n_campaigns": int(events["channel"].nunique()),
        "journey_summary": {
            "journey_count": journeys.summary.journey_count,
            "converter_count": journeys.summary.converter_count,
            "non_converter_count": journeys.summary.non_converter_count,
            "unique_path_count": journeys.summary.unique_path_count,
            "average_path_length": journeys.summary.average_path_length,
        },
        "timings_seconds": times,
        "memory": memory,
        "baseline_reliability": baseline.score.value,
        "baseline_components": dict(baseline.score.components),
        "identity_loss": identity_rows,
        "touchpoint_loss": touch_rows,
        "duplicates": duplicate_rows,
        "conversion_window": {
            "score": window.score.value,
            "message": window.findings[0].message if window.findings else "",
        },
        "contamination": {
            "score": contamination.score.value,
            "message": contamination.findings[0].message if contamination.findings else "",
            "details_available": bool(contamination.findings),
        },
        "model_shares_linear_top10": model_results["linear"].data.head(10).to_dict(orient="records"),
        "methodological_notes": [
            "Criteo has anonymized campaigns, not named media channels.",
            "The adapter marks one conversion event per conversion_id; it does not treat every impression-level conversion flag as a conversion.",
            "Timestamps are seconds from the first impression, converted with unit='s'.",
            "Attribution remains observational and does not establish incrementality.",
            sampling_note,
            "Markov removal effects are computed with a numpy transition matrix; "
            "all observed campaigns remain states unless MarkovAttribution(max_channels=...) "
            "is set explicitly.",
        ],
    }
    json_path = output_dir / "criteo_benchmark.json"
    md_path = output_dir / "criteo_benchmark.md"
    json_path.write_text(json.dumps(payload, indent=2, default=str))
    markdown = "\n".join(
        [
            "# Criteo reliability benchmark",
            "",
            sampling_note,
            "",
            "## Identity loss",
            "",
            "```text",
            _markdown_table(identity_rows),
            "```",
            "",
            "## Touchpoint loss",
            "",
            "```text",
            _markdown_table(touch_rows),
            "```",
            "",
            "## Duplicate-event detection",
            "",
            pd.DataFrame(duplicate_rows).to_string(index=False),
            "",
            "## Performance",
            "",
            json.dumps(times, indent=2),
            "",
            json.dumps(memory, indent=2),
            "",
            "## Delayed-converter contamination",
            "",
            contamination.findings[0].message if contamination.findings else "",
            "",
            "## Conversion-window sensitivity",
            "",
            window.findings[0].message if window.findings else "",
            "",
            "## Interpretation",
            "",
            "Duplicate injection is expected to move the data-quality score immediately.",
            "Identity and touchpoint loss are expected to show up first as journey-count",
            "changes, path-length changes, share drift, and rank correlation — not",
            "necessarily as a large move in the generic 0-100 heuristic unless the",
            "identity-robustness component is included in `checks`.",
            "",
            "## Limitations",
            "",
            "- Results are from the mapped Criteo sample described above.",
            "- Campaign IDs are anonymized; do not interpret them as publisher or channel names.",
            "- Simulated identity and touchpoint loss are synthetic probes, not estimates of Criteo's true identity graph.",
        ]
    )
    md_path.write_text(markdown + "\n")
    model_results["linear"].data.to_csv(output_dir / "criteo_linear_attribution.csv", index=False)
    print(markdown)
    print(f"\nWrote {json_path} and {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
