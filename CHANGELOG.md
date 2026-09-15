# Changelog

All notable changes follow [Keep a Changelog](https://keepachangelog.com/) conventions.

## 0.1.2 - 2026-09-15

### Changed

- Build source distributions without repository-local tooling files.

## 0.1.1 - 2026-09-15

### Fixed

- Constrain NumPy below 2.5 so its installed stubs remain parseable while mypy
  checks the supported Python 3.11 minimum from Python 3.12/3.13 environments.
- Make contributor setup and all CI matrix jobs use the same frozen `uv.lock`;
  run mypy on Python 3.11, 3.12, and 3.13 instead of hiding it in one job.

## 0.1.0 - 2026-09-15

### Added

- Add a standalone Python library for auditing observational multi-touch
  attribution reliability.
- Add typed event validation and converter/non-converter journey construction.
- Add first-touch, last-touch, linear, and absorbing Markov attribution.
- Add data-quality, duplicate, path, window, identity-loss, touchpoint-loss,
  temporal-stability, and model-disagreement checks.
- Add deterministic journey-level bootstrap attribution intervals and rank frequencies.
- Add decision robustness and rank/share assumption-sensitivity matrices.
- Add typed insufficient-evidence statuses that abstain from aggregate scoring.
- Add six deterministic known-condition stress scenarios.
- Add standalone Markdown and HTML reports.
- Add complete-user sampling with report-visible metadata.
- Add public Criteo and Hugging Face synthetic dataset adapters.
- Add independent JD-style and Criteo Research label-mismatch synthetic generators.
- Add a Poisson multi-touch synthesizer compatible with the PubliusV CSV schema.
- Add optional adapters for IgnazioDS journeys and triangulation user-level MTA tables.
- Identify Markov only when NULL/non-converter paths are observed.
- Score identity robustness in the default audit run.
- Treat date-grain repeated touches as informational, not duplicate failures.
- Fail data quality when converting events have zero conversion value.
- Add transparent configurable scoring weights, interpretation bands, and a
  reproducible score-sensitivity benchmark.
