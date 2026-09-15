# API stability

`mta-audit` 0.1.x is **alpha**. The public surface below is what release
reviews treat as the library. Changes that alter scores or rename these
symbols will be called out in the changelog.

## Supported

- `MTAAudit` construction and `run()`
- `AuditReport` (`summary`, `to_dataframe`, `to_json`, `to_markdown`,
  `to_html`, `decision_robustness`, `sensitivity_matrix`, score components,
  `bootstrap`, `markov_identified`, `channel_grain`, `sample_info`)
- `BootstrapConfig`, `BootstrapResult`, `DecisionRobustnessResult`,
  `SensitivityResult`, and `EvidenceStatus`
- Built-in models: `first_touch`, `last_touch`, `linear`, `markov`
- `load_criteo_attribution` / `CriteoAttributionAdapter`
- `load_synthetic_attribution` / `transform_synthetic_attribution`
- `simulate_jd_mta_events`, `simulate_criteo_label_events`,
  `simulate_synthesizer_events`
- `transform_synthesizer_touchpoints`, `transform_ignazio_touchpoints`,
  `transform_triangulation_mta`, and optional loaders for those CSVs
- `ColumnMapping`, `validate_events`, `build_journeys`
- `sample_for_audit` and deterministic public stress scenarios
- Six-component scoring weights and default interpretation bands

## Not supported as a stable contract

- Private helpers, benchmark scripts, and notebook cells
- Warehouse-specific extracts (not part of this repository)
- Optional plot layouts (`viz` extra)

## Scoring policy

Default check membership and scoring formulas are part of the 0.1 contract.
Changing a default weight, adding a default check, or changing how a
component is aggregated is a changelog-worthy behavior change. Skipped
components still do not count as 100.
