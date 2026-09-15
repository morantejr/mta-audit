# Criteo reliability benchmark

Explicit representative sample of **8,000 users** (`random_state=42`) from the
public Criteo Attribution Modeling file (16.5M impressions, ~700 campaigns).
This is not silent downsampling. The local extract had **21,307 mapped events**,
**639 campaigns**, **8,144 journeys**, **586 conversions**.

Numbers below are from `benchmarks/criteo_benchmark.py` on 2026-09-14.
They are not illustrative placeholders.

## Identity loss

Identity fragmentation increased journey count (8,144 → 9,290 at 30%) and
shortened average path length (2.49 → 2.20). Linear-share total-variation
drift peaked at **0.010** (20% rate) and Spearman rank correlation of campaign
shares fell to **0.983**. The headline 0–100 heuristic stayed near 72 because
this check subset does not include the identity-robustness simulation
component; detection is in the drift / fragmentation columns.

```text
Scenario                  Rel   Drift  RankCorr  Journeys  UniquePaths
---------------------------------------------------------------------
Baseline                72.35  0.0000    1.0000      8144         3198
5% identity loss        72.36  0.0000    0.9989      8335         3187
10% identity loss       72.37  0.0032    0.9977      8526         3161
20% identity loss       72.28  0.0102    0.9833      8908         3126
30% identity loss       72.43  0.0048    0.9924      9290         3120
```

## Touchpoint loss

Randomly dropping non-conversion impressions moved linear campaign ranks
(`rank_corr` 1.00 → 0.97 at 30%) and increased share drift to **0.022**.
Journey count fell because some users lost all remaining touches in-window.

```text
Scenario                  Rel   Drift  RankCorr  Journeys  UniquePaths
---------------------------------------------------------------------
Baseline                72.35  0.0000    1.0000      8144         3198
5% touchpoint loss      72.33  0.0039    0.9900      7943         3102
10% touchpoint loss     72.61  0.0077    0.9862      7709         2983
20% touchpoint loss     72.95  0.0157    0.9755      7261         2743
30% touchpoint loss     73.16  0.0216    0.9723      6727         2509
```

## Duplicate-event detection

Injected duplicates were **detected in every scenario**. Restricting the audit
to data-quality / duplicate checks dropped the reported score from ~72 to ~50.

```text
scenario               reliability  duplicates_detected  rows_injected
1% duplicate events          50.70                 True            213
5% duplicate events          50.50                 True           1065
10% duplicate events         50.20                 True           2131
```

## Conversion-window sensitivity

Attribution across 1/3/7/14/30-day windows had minimum rank correlation
**0.588** and maximum share volatility **2.9%**. Campaign rankings are
sensitive to the conversion window even when share mass does not swing
wildly — budget conclusions that depend on rank should be treated cautiously.

## Delayed-converter contamination

Using first impression to stored `conversion_timestamp` (Criteo's 30-day
horizon): **180 of 8,000 users (2.2%)** convert after a 7-day short window
but within 30 days. Those users would be labeled non-converters if
observation stopped at day 7.

## Performance (this run)

| Step | Seconds |
| --- | --- |
| Load + map Criteo file | 19.4 |
| Journey construction | 1.3 |
| First/last/linear/Markov | 5.2 |
| Baseline multi-check audit | 24.8 |

Peak process RSS was about **2.9 GB**, dominated by reading the 623 MB gzip
into pandas before user sampling. Markov used all 639 campaign states via a
numpy removal-effect implementation (`MarkovAttribution(max_channels=N)` is
available if you want an explicit long-tail collapse).

## Top linear-share campaigns (anonymized IDs)

See `criteo_linear_attribution.csv`. The largest linear share in this sample
was campaign `15184511` at about **5.8%** of conversion credit. Do not map
these IDs to publisher or channel names.

## Does mta-audit detect known problems?

- **Duplicates:** yes — injected rows were flagged and the quality score fell.
- **Identity loss:** yes as fragmentation and rank/share drift; the generic
  heuristic score barely moved because identity-robustness was not one of the
  scored components in the corruption grid.
- **Touchpoint loss:** yes as attribution drift and fewer/shorter journeys.
- **Window choice:** yes — Spearman 0.59 across windows is a high-sensitivity
  result even on this sample.

## Limitations

- Attribution is observational and does not establish incrementality.
- Campaign IDs are anonymized; this is not a named-media-channel study.
- Identity/touchpoint corruptions are synthetic probes, not Criteo's true
  identity graph or pixel-loss process.
- The 30-day file cannot observe conversions after day 30.
