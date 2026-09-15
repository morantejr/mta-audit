# API reference

## `MTAAudit`

`MTAAudit(data=..., columns=..., lookback_window="30D",
conversion_window=None, scoring_weights=None)` stores immutable configuration.
Individual `user_col`, `timestamp_col`, `channel_col`, `conversion_col`,
`conversion_value_col`, `conversion_timestamp_col`, `campaign_id_col`
(`campaign_col` remains an alias), `impression_col`, `click_col`, `cost_col`,
`revenue_col`, `device_col`, `publisher_col`, `creative_id_col`, and
`conversion_id_col`/`order_id_col` arguments are alternatives to `columns`. Normalization preserves
unmapped source columns for diagnostics and corruption workflows.

`run(events=None, checks=None, simulations=None, attribution_models=None,
conversion_windows=None, bootstrap=False, n_bootstrap=200, random_state=42)`
builds journeys and returns `AuditReport`. `bootstrap` accepts `True` or a
typed `BootstrapConfig`. Built-in
model names are `first_touch`, `last_touch`, `linear`, `markov`, and
`markov_chain`.

Individual methods include `check_conversion_window`,
`check_delayed_converter_contamination`, `check_model_stability`,
`simulate_identity_loss`, `simulate_touchpoint_loss`, and
`simulate_event_loss`.
Mapped boolean impression and click columns let `simulate_event_loss` operate
without an `event_type_col`; typed event streams remain supported.

## `AuditReport`

- `summary()` returns a text scorecard and deduplicated recommendations.
- `to_dataframe()` returns one row per audit finding.
- `to_json(include_bootstrap_draws=False)` serializes nested dataclasses,
  pandas objects, timestamps, and numpy scalars. Bootstrap draws are omitted
  by default to keep reports bounded; opt in when replicate-level JSON is needed.
- `to_markdown()` and `to_html()` render standalone stakeholder reports.
- `sensitivity_matrix(metric="rank" | "share")` aligns channels across tested
  assumptions.
- `decision_robustness()` returns typed channel-level rank and decision
  stability.
- `plot_scorecard()`, `plot_model_comparison()`,
  `plot_window_sensitivity()`, and `plot_channel_volatility()` return matplotlib
  axes and require the `viz` extra.

`run(...)` now includes identity and touchpoint robustness plus
`markov_identification` by default. `AuditReport.markov_identified` is true
only when non-converter/NULL paths exist. `channel_grain` is `campaign` when
more than 50 labels appear.

`BootstrapResult.to_dataframe()` returns mean/median share, percentile
intervals, rank intervals, and top-k frequencies. Its `draws` table is
available for transparent local analysis. `status="insufficient_evidence"`
means the package abstained.

`sample_for_audit(data, max_journeys=100_000, stratify_by="conversion",
random_state=42)` selects complete user histories and stores sampling metadata
in the returned frame; `AuditReport.sample_info` exposes it.

`make_stress_scenario(name, random_state=42)` provides six documented,
deterministic known-condition observational fixtures.

`load_synthetic_attribution` is public. Independent research-style synthetics
are `simulate_jd_mta_events`, `simulate_criteo_label_events`, and
`simulate_synthesizer_events`. CSV adapters map IgnazioDS and triangulation
user-level tables; they do not vendor those projects' generators. See
[stability](stability.md) for the 0.1 supported surface and
[releasing](releasing.md) for install and PyPI.

`AuditFinding.metric_value` is the documented metric field. The legacy `value`
attribute and serialized key remain synchronized for compatibility.

## Lower-level APIs

`build_journeys` / `JourneyBuilder`, attribution model classes, audit functions, threshold
dataclasses, `corrupt`, `detect_duplicates`, `deduplicate`, and
`load_criteo_attribution` / `CriteoAttributionAdapter` are public.

Criteo timestamps are relative seconds from the first impression. The adapter
constructs conversion events from `conversion_id` and `conversion_timestamp`;
it does not treat the impression-level `conversion` flag as an event conversion.
Use `pytest -m criteo` for the optional integration suite.
