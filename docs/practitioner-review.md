# Practitioner review

The project needs evidence from people who did not build it. A useful review
should be based on an actual install and audit, not a README impression.

## Twenty-minute review

1. Install the latest tagged release in a clean Python 3.11+ environment.
2. Run `MTAAudit(...).run()` on either:
   - a small, de-identified event extract you are permitted to use, or
   - `make_stress_scenario("duplicate_contamination").events`.
3. Read `report.summary()` and the rendered HTML report.
4. Comment on the public review issue with:
   - whether installation and the quickstart worked;
   - which finding was actionable or confusing;
   - whether the component scores and aggregate were distinguishable;
   - one missing diagnostic or misleading claim;
   - whether you would use the output in an attribution review, and why.

Do not upload company, advertiser, user, or warehouse data. Aggregate feedback
and synthetic reproductions are sufficient.

## Acceptance evidence

The first external review is complete only when a person unaffiliated with the
implementation posts reproducible feedback. Automated-agent review, CI, stars,
and the maintainer's own benchmark do not count.
