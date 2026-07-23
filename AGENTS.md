# Repository Guidelines

## Project

- This repository contains Python User-Defined Table Functions (UDTFs) for PySpark.
- Use Python 3.10 or newer, PySpark 4.0 or newer, `uv` for dependency management, and Hatchling for builds.
- Source code lives in `src/pyspark_udtf/`, tests in `tests/`, examples in `examples/`, and documentation in `docs/`.

## Development workflow

- Install development dependencies with `uv sync --extra dev`.
- Add runtime dependencies with `uv add <package>` and development dependencies with `uv add --dev <package>`.
- Run tests with `uv run pytest`.
- Run lint checks with `uv run ruff check .` and format code with `uv run ruff format .`.
- Build distributions with `uv build`.

## Coding conventions

- Use type hints where practical and follow the Ruff configuration in `pyproject.toml`.
- Follow PySpark UDTF conventions for `eval`, `terminate`, and `analyze`. Implement `analyze` only when the output schema must be determined dynamically.
- Prefer `TABLE` arguments for table or partition processing over row-by-row `LATERAL` invocation when possible.
- Keep a UDTF self-contained unless logic is genuinely shared or complex enough to justify a utility module.
- Use the `$create-udtf` skill when creating a new UDTF.

## Testing conventions

- Use `pytest` and `pytest.fixture`; use `unittest.mock` for external dependencies such as HTTP requests.
- Test UDTF classes directly when possible to avoid unnecessary Spark startup overhead.
- Use `pyspark.sql.types.Row` to simulate input rows.
- Cover buffering behavior, including `eval` buffering and `terminate` flushing, plus relevant error paths and output schemas.
- Add or update tests for every behavior change, then run the focused tests followed by the full suite and Ruff checks.
