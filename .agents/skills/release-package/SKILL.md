---
name: release-package
description: Release a new version of the pyspark-udtf package by checking the repository, rebasing, bumping the version, building distributions, committing, publishing to PyPI, and pushing the release commit. Use only when the user explicitly invokes $release-package.
---

# Release Package

Release a requested version of `pyspark-udtf` sequentially. Never print or expose PyPI credential values.

## Prerequisites

- Require `uv` 0.7.0 or newer when possible.
- Require a clean Git working tree.
- Confirm that PyPI credentials are configured or that `UV_PUBLISH_TOKEN` is present without displaying its value.

## Workflow

### 1. Verify the environment

Run `uv --version` and inspect `git status --porcelain`. Stop if the working tree is not clean.

### 2. Rebase on master

Fetch `origin` and rebase the current branch onto `origin/master`:

```bash
git fetch origin
git rebase origin/master
```

If conflicts occur, run `git rebase --abort`, report the conflict, and stop. Do not resolve rebase conflicts automatically.

### 3. Bump the version

Read the current version and apply the requested patch, minor, major, or exact version change.

- Prefer `uv version --bump <type>` with supported `uv` versions.
- Otherwise update `pyproject.toml` manually.
- Confirm that the resolved version matches the user's request.

### 4. Build the package

Run `uv build`. Verify that `dist/` contains both the new `.tar.gz` source distribution and `.whl` wheel, and that their filenames contain the requested version.

### 5. Commit the version bump

Stage only the intended version files and commit them with `Bump version to X.Y.Z`.

### 6. Publish to PyPI

Before running `uv publish`, show the user the exact version and distribution filenames and get explicit confirmation to publish them to PyPI. Stop without publishing if confirmation is not provided.

After confirmation, run `uv publish` and verify that the requested version is available on PyPI.

### 7. Push the release commit

- If no history rewrite is required, run `git push`.
- If the rebase requires `git push --force-with-lease`, show the user the remote and branch and get explicit confirmation before running it.
- Never use an unconditional force push.

## Verification

- Confirm that `dist/` contains only the intended version artifacts used for this release.
- Confirm that the release commit is present on the expected remote branch.
- Confirm that the new version is available on PyPI.
