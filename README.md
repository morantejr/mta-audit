# mta-audit

**Stress-testing and reliability testing for multi-touch attribution.**

`mta-audit` evaluates whether campaign conclusions remain stable when
attribution windows, identity quality, touchpoint availability, attribution
models, sampling, and other measurement assumptions change.

> It does not estimate causal lift or incrementality.

This project is a standalone reliability library. Attribution systems still
allocate credit; this package scores whether journeys and model outputs are
stable enough to use.

Optional public-dataset adapters only map foreign tables onto the package event
schema.

## What it catches

The public Criteo benchmark uses an explicit 8,000-user sample: 21,307 mapped
events, 639 anonymized campaigns, 8,144 journeys, and 586 conversions.

- Injecting 1–10% duplicate events was detected every time and reduced the
  scoped data-quality score from **72.35 to 50.20–50.70**.
- Conversion-window campaign ranks reached a minimum Spearman correlation of
  **0.588** across 1/3/7/14/30-day windows.
- Dropping 30% of eligible touchpoints produced **2.2% attribution-share
  drift** and reduced campaign rank correlation to **0.972**.
- Fragmenting 20% of identities increased journeys from **8,144 to 8,908**.

See the [rendered sample audit](benchmarks/results/sample_report.html), its
[Markdown source](benchmarks/results/sample_report.md), and the full
[Criteo reliability scorecard](benchmarks/results/criteo_benchmark.md).
The sample is synthetic and deterministic; the Criteo figures are measured,
not illustrative.

## Installation

Python 3.11–3.13 is supported.

```bash
pip install mta-audit
pip install "mta-audit[viz]"
```

For an unreleased main-branch snapshot:

```bash
pip install "mta-audit @ git+https://github.com/morantejr/mta-audit.git"
```

For contributors, use the committed lockfile:

```bash
uv sync --frozen --extra dev
uv run --no-sync pytest
uv run --no-sync ruff check src tests
uv run --no-sync mypy
```

CI uses the same commands and lock resolution on Python 3.11, 3.12, and 3.13.
Do not use an unconstrained `pip install -e ".[dev]"` as the reproducibility
check; that intentionally resolves a different environment.

## Quickstart

```python
import pandas as pd
from mta_audit import MTAAudit

data = pd.DataFrame({
    "user_id": ["a", "a", "b", "b"],
    "timestamp": pd.to_datetime([
        "2026-01-01", "2026-01-03", "2026-01-02", "2026-01-04"
    ]),
    "channel": ["search", "email", "social", "direct"],
    "conversion": [False, True, False, True],
})

report = MTAAudit(data=data).run(
    bootstrap=True,
    n_bootstrap=200,
    random_state=42,
)
print(report.summary())
print(report.decision_robustness())
report.sensitivity_matrix(metric="rank")
report.to_dataframe()
report.to_json()
```

Example output:

```text
MTA Audit Reliability Report
Overall audit score: 67.5/100 — Fragile
Component scores:
- Data Quality: …
- Window Stability: …
- Identity Robustness: …
- Path Quality: …
- Model Agreement: …
- Temporal Stability: insufficient evidence
Checks run: 12 | Visible findings: 7
Markov removal effects: not evaluated
Channel grain: channel

Decision robustness:
Search remains top-two in 94% of tested scenarios.
Display and Social frequently change position; avoid a strong reallocation
between them based on this observational attribution analysis alone.
```

Custom schemas and the advanced API remain concise:

```python
audit = MTAAudit(
    data=data,
    user_col="visitor_id",
    timestamp_col="event_time",
    channel_col="source",
    conversion_col="converted",
    scoring_weights={"data_quality": 2, "model_agreement": 1},
)
report = audit.run(
    attribution_models=["first_touch", "last_touch", "linear", "markov"],
    conversion_windows=[7, 14, 30, 60],
    checks=["data_quality", "conversion_window", "model_disagreement"],
    simulations={"identity_loss": {"rates": [0.05, 0.10, 0.20]}},
    bootstrap=True,
    n_bootstrap=500,
    random_state=42,
)
```

For reusable bootstrap settings:

```python
from mta_audit import BootstrapConfig

report = MTAAudit(data).run(
    bootstrap=BootstrapConfig(
        n_iterations=500,
        confidence_level=0.90,
        random_state=42,
    )
)
report.bootstrap.to_dataframe()
```

Bootstrap resamples complete journeys, never touchpoint rows. Its percentile
intervals and rank frequencies measure observational attribution stability;
they are not causal confidence intervals.

## Checks

- Required fields, nulls, timestamps, channels, and duplicate events
- Conversion-window sensitivity and delayed-converter contamination
- First-touch, last-touch, linear, and absorbing Markov model disagreement
- Path sparsity and channel concentration
- Weekly or monthly temporal stability
- Converter/non-converter exposure comparison
- Deterministic identity, touchpoint, impression, and click-loss simulations

## Methodology

Every audit emits a normalized 0–100 **audit robustness score**, severity,
observed metric, and recommendation. The score does not mean percent correct.
Related checks feed six reliability components:
`data_quality`, `window_stability`, `identity_robustness`, `path_quality`,
`model_agreement`, and `temporal_stability`. The overall formula is:

```text
reliability = Σ(component score × normalized available-component weight)
```

Only completed components are included; skipped checks are never treated as
perfect. Attribution models allocate observed conversion value. Markov credit
uses channel-removal effects from an absorbing transition chain.

Qualitative bands are Robust (90–100), Stable (80–89), Caution (70–79),
Fragile (60–69), and Highly Fragile (<60). The component scorecard is more
important than the aggregate.

## Shareable reports and local sampling

```python
markdown = report.to_markdown()
html = report.to_html()

from mta_audit import sample_for_audit
sample = sample_for_audit(data, max_journeys=100_000, random_state=42)
sampled_report = MTAAudit(sample).run()
print(sampled_report.sample_info)
```

Sampling preserves complete user histories and marks the resulting report.
This remains a local pandas package; no warehouse or distributed execution is
included.

## Thresholds and configuration

Default scoring weights are 20%, 15%, 15%, 15%, 20%, and 15% in the component
order above. Any non-negative relative values can be passed through
`scoring_weights`; they are normalized automatically over available components.
These are explicit policy defaults, not empirically estimated constants:
data quality and model agreement receive 20% because invalid inputs and
conclusion-changing model choice are direct decision risks; the other four
dimensions receive equal 15% weights. The round-number bands are likewise
communication thresholds, not confidence intervals.

The reproducible [score-policy sensitivity report](benchmarks/results/scoring_sensitivity.md)
reweights the committed Criteo components and shifts every rating cutoff by
five points. It shows how much the aggregate and label depend on those policy
choices. Component scores must remain visible; users should override weights
when their decision context differs.
Audit-specific threshold dataclasses such as `ConversionWindowThresholds`,
`ContaminationThresholds`, and `TemporalStabilityThresholds` expose warning
cutoffs. See [the methodology](docs/methodology.md) for formulas.

## Visualization

With the `viz` extra installed, reports provide `plot_scorecard()`,
`plot_model_comparison()`, `plot_window_sensitivity()`, and
`plot_channel_volatility()`. Each returns a matplotlib `Axes`.

## Limitations

Results depend on identity resolution, channel taxonomy, event collection, and
the selected lookback/conversion windows. Sparse paths make transition estimates
unstable. Converter/non-converter comparisons are confounded and are not lift
estimates. Tracking-loss simulations test sensitivity to specified corruption,
not the true unknown missingness process.

Converter-only sequencing tables do **not** identify Markov removal effects.
The report sets `markov_identified=false` in that case. Date-grain
timestamps are not treated as duplicates. Campaign-name "channels" are flagged as
fine grain.

Kaggle does not currently publish a licensed event-level multi-touch path dataset
with converters and non-converters comparable to Criteo. Public fixtures:

- Criteo Attribution Modeling (real impressions; see walkthrough below)
- Hugging Face [synthetic attribution benchmark](https://huggingface.co/datasets/lucianfialho/synthetic-attribution-benchmark) (CC BY 4.0)
- Independent generators inspired by JD MTA (Du et al., 2019) and Criteo Research
  robust-label experiments (Bompaire et al., 2020): `simulate_jd_mta_events`,
  `simulate_criteo_label_events`, `simulate_synthesizer_events`
- Optional CSV adapters for IgnazioDS journeys and the triangulation project's
  user-level `df_mta.csv` (`load_ignazio_attribution`, `load_triangulation_mta`).
  Those files are not stored in git. Weekly MMM / geo tables are out of scope.

## Roadmap

Current stage: **0.1.0 alpha**. Next:

1. Keep 0.1.x installable and documented (stability contract in
   [docs/stability.md](docs/stability.md)).
2. **Beta** when `run()` / report fields stay boring across a few tagged
   releases and the Criteo reference scorecard is re-run before each tag.
3. **1.0** when scoring defaults are frozen and third parties can depend on
   the public API. Warehouse extracts stay out of this repository.

Later, optional work: more public-dataset adapters and richer performance
benchmarks.

See [concepts](docs/concepts.md), [methodology](docs/methodology.md),
[API reference](docs/api.md), [stability](docs/stability.md),
[releasing](docs/releasing.md), the
[external practitioner review protocol](docs/practitioner-review.md), and
[examples](examples/quickstart.py).

## Criteo walkthrough

The public Criteo Attribution Modeling dataset (CC-BY-NC-SA-4.0) is the
real-world integration set for this package. **It is not stored in git.**

1. Download `criteo_attribution_dataset.tsv.gz` from
   [Hugging Face](https://huggingface.co/datasets/criteo/criteo-attribution-dataset)
   (Criteo's published copy of Diemert et al., 2017).
2. Save it as `./data/criteo_attribution_dataset.tsv.gz`, or let the loader
   cache it under `~/.cache/mta-audit/` with `download=True`.
3. Map impressions through the adapter — do **not** pass Criteo's impression-level
   `conversion` flag straight into `conversion_col`.

```python
from mta_audit import MTAAudit
from mta_audit.datasets import load_criteo_attribution, CriteoAttributionAdapter

# Explicit representative sample; pass sample_users=None for the full file.
df = load_criteo_attribution(path="./data/criteo_attribution_dataset.tsv.gz", sample_users=20000)
# equivalent: CriteoAttributionAdapter("./data/criteo_attribution_dataset.tsv.gz").load()

report = MTAAudit(data=df).run(
    attribution_models=["first_touch", "last_touch", "linear", "markov"]
)
print(report.summary())
```

Public synthetic journeys (named channels + NULL paths, CC BY 4.0):

```python
from mta_audit.datasets import load_synthetic_attribution

events = load_synthetic_attribution(download=True)
report = MTAAudit(data=events).run(attribution_models=["linear", "markov"])
print(report.markov_identified)
```

Independent research-style synthetics (no third-party generators vendored):

```python
from mta_audit.datasets import (
    simulate_criteo_label_events,
    simulate_jd_mta_events,
    simulate_synthesizer_events,
)

jd = simulate_jd_mta_events(n_users=200, random_state=42)
labels = simulate_criteo_label_events(n_users=400, random_state=42)
synth = simulate_synthesizer_events(n_users=300, random_state=42)
```

Unit tests never require the file. Integration tests and the benchmark do:

```bash
pytest                    # default, no Criteo download
pytest -m criteo          # loads the local/public dataset
python benchmarks/criteo_benchmark.py --download --sample-users 20000
```

Results from a completed run (8,000-user explicit sample) live in
[`benchmarks/results/criteo_benchmark.md`](benchmarks/results/criteo_benchmark.md).
On that run, injected duplicates dropped the data-quality score from ~72 to ~50,
20% identity loss raised journey count from 8,144 to 8,908, and conversion-window
ranks correlated at only 0.59. The notebook `examples/criteo_example.ipynb` is
the measurement-team walkthrough.

Criteo campaigns are anonymized. Treat outputs as campaign-level attribution
reliability diagnostics, not named-channel budget recommendations.

Removal-effect Markov on ~700 campaigns is implemented with a numpy transition
matrix. `MarkovAttribution(max_channels=N)` can explicitly collapse a long tail
into `__OTHER__`; the benchmark does **not** do that by default.
