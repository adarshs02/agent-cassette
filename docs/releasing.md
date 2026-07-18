# Releasing

Publishing is automated by `.github/workflows/publish.yml` using PyPI
[Trusted Publishing](https://docs.pypi.org/trusted-publishers/) (OIDC) — no API
tokens are stored.

- **PyPI** — publishes automatically when a GitHub Release is *published*.
- **TestPyPI** — publishes on a manual `workflow_dispatch` run (dry-run/staging).

The workflow builds the wheel and sdist with `uv build`, runs `twine check`, then
publishes from the built artifact via `pypa/gh-action-pypi-publish`.

## One-time setup (per PyPI account, done on the web)

Trusted Publishing must be configured once on each index before the first publish.
For a project that does not exist yet, add a **pending publisher**.

On [pypi.org](https://pypi.org/manage/account/publishing/) and
[test.pypi.org](https://test.pypi.org/manage/account/publishing/), add a GitHub
publisher with:

| Field | Value |
| --- | --- |
| PyPI Project Name | `agent-cassette` |
| Owner | `adarshs02` |
| Repository name | `agent-cassette` |
| Workflow name | `publish.yml` |
| Environment name | `pypi` (on PyPI) / `testpypi` (on TestPyPI) |

The GitHub environments `pypi` and `testpypi` already exist in repository settings.

## Cutting a release

1. Land all changes on `main` with CI green.
2. Bump `version` in `pyproject.toml`, update `CHANGELOG.md`, refresh `uv.lock`
   (`uv lock`), and merge.
3. Tag and create the GitHub Release:
   ```bash
   git tag -a vX.Y.Z -m "agent-cassette X.Y.Z"
   git push origin vX.Y.Z
   gh release create vX.Y.Z --target main --title "vX.Y.Z" --notes-file <notes>
   ```
   Publishing the release triggers the PyPI upload.

## TestPyPI dry-run

Run the workflow manually to publish the current build to TestPyPI:

```bash
gh workflow run publish.yml
```
