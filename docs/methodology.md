# Methodology

## Reliability score

Each completed audit returns a 0–100 **audit robustness score**. It is an
index of observed stability under the checks that ran, not a probability of
correctness. Related checks are averaged within their component:

- data quality: schema/data checks and duplicates
- window stability: conversion-window sensitivity
- identity robustness: identity, touchpoint, or typed event loss
- path quality: path sparsity and channel concentration
- model agreement: pairwise model disagreement
- temporal stability: period-to-period model stability

If `s_c` is a component score and `w_c` its configured weight, the report score is
`sum(s_c * w_c / sum(available w))`. Thus an unrun audit contributes neither 100
nor zero. `AuditScore.components` and `AuditScore.weights` expose every input.

Default relative weights are 0.20, 0.15, 0.15, 0.15, 0.20, and 0.15,
respectively. Default interpretation bands are Robust ≥90, Stable ≥80,
Caution ≥70, Fragile ≥60, and Highly Fragile below 60.

`AuditReport.markov_identified` is true only when a Markov model is requested
**and** at least one non-converter (NULL) path is observed. Converter-only
tables still produce first/last/linear shares; they do not identify removal
effects. `channel_grain` is `campaign` when more than 50 distinct labels
appear; path quality then flags fine grain so campaign or creative names are
not treated as media channels.

## Audit formulas

Window stability combines the worst pairwise rank correlation (mapped from
[-1, 1] to [0, 1]) with one minus maximum channel-share range. Model agreement
averages rank correlation, Jensen–Shannon similarity, maximum-share similarity,
and normalized rank stability. Path quality penalizes singleton path share and
channel HHI. Temporal stability compares share/rank ranges, conversion-rate
drift, and Markov transition similarity where available.

Tracking robustness reruns models after deterministic corruption and penalizes
maximum channel-share drift and model instability. Threshold dataclasses are
public so production policy can be explicit and version-controlled.

## Bootstrap uncertainty

Bootstrap uncertainty resamples complete built journeys with replacement. For
each requested attribution model and channel it reports mean/median share,
standard deviation, a configurable percentile interval, median/percentile
rank, and frequencies of ranking in the top 1, 2, or 3. By default converters
and NULL paths are resampled separately, preserving their observed counts and
preventing a bootstrap draw from accidentally making Markov unidentified.

If there are fewer than three journeys or fewer than the configured minimum
converted journeys, the result has `status="insufficient_evidence"` and no
statistics. Missing evidence is not scored as perfect.

These are percentile intervals for the attribution procedure applied to the
observed journey population. They are not confidence intervals for causal lift.

## Decision robustness and sensitivity

The sensitivity matrix aligns attribution share or rank across outputs already
computed by the audit: baseline, attribution models, conversion windows,
temporal periods, and tracking-loss scenarios. It does not invent unexecuted
assumptions.

Decision robustness reports assumption-scenario and bootstrap rank behavior
separately so hundreds of bootstrap draws cannot outweigh a few substantive
measurement assumptions. For each channel it reports baseline and median rank,
5th–95th percentile rank, top-k frequency, share range, rank standard deviation,
separate bootstrap rank fields, and an interpretable classification based only
on assumption scenarios:

- Robust: top-one frequency ≥75%, top-two frequency ≥90%, and rank span ≤1
- Stable: top-one frequency ≥50%, top-two frequency ≥75%, and rank span ≤2
- Uncertain: top-two frequency ≥50%
- Fragile: otherwise

For two-channel results, where every channel is tautologically top-two, the
classification uses top-one frequency instead.

These frequencies are conditional on tested assumptions. They are not
probabilities of causal superiority.

## Interpretation

Scores are diagnostics, not uncertainty intervals or causal evidence. Use them to
identify assumptions that deserve investigation and to compare reliability under
the same configuration.

Diagnostics abstain with `insufficient_evidence` when the required variation is
absent—for example, one attribution model, one time period, one conversion
window, or too few converted journeys for bootstrap estimation. Such checks are
excluded from the weighted score and remain visible in the report.

## Score policy and sensitivity

The default component weights are a review policy, not learned parameters:
20% data quality, 15% window stability, 15% identity robustness, 15% path
quality, 20% model agreement, and 15% temporal stability. Data quality and
model agreement receive modest extra weight because bad inputs invalidate all
downstream analysis and model choice can reverse the decision. The other
dimensions are equally weighted because there is no defensible universal
ordering across organizations.

Weights are renormalized over evaluated components. This avoids treating
missing evidence as perfect evidence, but means two reports with different
available components are not strictly like-for-like. Always show the component
scorecard with the aggregate.

The 90/80/70/60 interpretation cutoffs are round communication bands, not
estimated error probabilities. The committed
[sensitivity report](../benchmarks/results/scoring_sensitivity.md) applies
equal, data-quality-heavy, and model-agreement-heavy policies to the Criteo
benchmark and shifts all label boundaries by ±5 points. Reproduce it with:

```bash
python benchmarks/scoring_sensitivity.py
```

If a conclusion changes under reasonable policies, report that dependence
instead of selecting the most favorable aggregate.

## Criteo adapter

The public Criteo file is impression-level. Using `conversion` as
`conversion_col` would count almost every impression on a converting user as a
conversion. The adapter instead marks one conversion per `(uid, conversion_id)`
on the last impression at or before `conversion_timestamp`, and stores the true
outcome time in `conversion_timestamp` so windows and delayed-converter checks
use time-to-convert rather than time-to-last-impression.

Campaign identifiers are mapped to `channel` because the dataset has no named
media taxonomy. Delayed-converter contamination on Criteo is therefore: users
whose first impression-to-conversion delay exceeds the short window but is
still within 30 days (the dataset's own horizon). Conversions after day 30 are
not in the file.

Reliability scores on Criteo use the same heuristics as any other dataset. They
are not a claim about Criteo-internal model quality.

## External synthetics

`simulate_jd_mta_events` follows the published sequential DGP in Du et al.
(2019) at test scale. The GPL-2.0 TensorFlow generator and tfrecord are not
copied into this package.

`simulate_criteo_label_events` follows Bompaire et al. (2020): type-B displays
need not trigger conversion, but last-touch still labels the last display.
Use it to check whether model-disagreement diagnostics react to that labeling
problem. It is not a bid-optimization or causal-lift API.

`simulate_synthesizer_events` is an independent Poisson / weighted-channel
process. `transform_ignazio_touchpoints` collapses journey-level `converted`
flags onto the last touch. `transform_triangulation_mta` accepts only the
user-level touch table; weekly MMM and geo-holdout files are out of scope.
