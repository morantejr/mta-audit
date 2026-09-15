# Score-policy sensitivity

This is a deterministic policy sensitivity check over the committed Criteo
baseline components. It does not recalibrate component checks or claim that
one policy is objectively correct.

Available Criteo components in this benchmark run:
- `data_quality`: **59.25**
- `path_quality`: **61.81**
- `model_agreement`: **93.34**

## Weight sensitivity

- **default: 72.35**
- **equal: 71.47**
- **data-quality-heavy: 64.49**
- **model-agreement-heavy: 83.97**

Policy-weight spread: **19.48 points**. The default result
(72.35) is not invariant to stakeholder priorities;
the component scores must therefore remain visible beside it.

## Interpretation-band sensitivity

- **default: Caution** ((90, 80, 70, 60))
- **five-points-stricter: Fragile** ((95, 85, 75, 65))
- **five-points-looser: Caution** ((85, 75, 65, 55))

The qualitative label can change while the measured component evidence
does not. Labels are communication policy, not statistical confidence.

Reproduce with `python benchmarks/scoring_sensitivity.py`.
