# Releasing

## Installable alpha

```bash
pip install mta-audit
```

## Tag a version

1. CI is green on `main`.
2. Version in `pyproject.toml` and `src/mta_audit/__init__.py` matches the tag.
3. Changelog section exists for that version.
4. Push an annotated tag matching the package version, for example:
   `git tag -a v0.1.1 -m "mta-audit v0.1.1" && git push origin v0.1.1`.

## PyPI (Trusted Publishing)

The `Publish` workflow builds a wheel and uploads it with OIDC when a `v*`
tag is pushed. No API token is stored anywhere. The publisher registered on
PyPI is:

- Owner: `morantejr`
- Repository: `mta-audit`
- Workflow: `publish.yml`
- Environment: `pypi`

All four claims must match exactly, including the environment. A mismatch
fails the upload with `invalid-publisher` after a successful build, which is a
PyPI-side configuration error rather than a packaging error. The failure output
prints the claims the workflow actually presented; compare those against the
publisher entry before changing anything else.

## After 0.1

- **Beta:** no breaking `run()` / report field renames without a changelog
  `Removed`/`Changed` note; re-run the Criteo benchmark script locally and
  refresh `benchmarks/results/` if numbers move.
- **1.0:** same public API plus a stated scoring freeze. Do not put private
  warehouse tables or advertiser extracts in git.
