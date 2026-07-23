# Contributing to pyspark-udtf

Thanks for helping improve the Python UDTF examples in this repository. Bug fixes, documentation improvements, new examples, and reusable UDTFs are welcome.

## Before you start

- Search the existing issues before opening a new one.
- Open an issue before making a large behavioral change or introducing a new dependency.
- Never include API tokens, credentials, customer data, or other secrets in code, tests, logs, or screenshots.
- Follow the [Code of Conduct](CODE_OF_CONDUCT.md) in all project spaces.

## Development setup

The project requires Python 3.10 or later and uses [uv](https://docs.astral.sh/uv/) for dependency management.

```bash
git clone https://github.com/allisonwang-db/pyspark-udtf.git
cd pyspark-udtf
uv sync --extra dev
```

Run the test and lint suites before submitting a pull request:

```bash
uv run pytest
uv run ruff check .
uv run ruff format --check .
```

Some integration tests call external services and require credentials. Keep those credentials in environment variables, never commit them, and describe any tests you could not run in the pull request.

## Adding a Python UDTF

1. Create a design document in `docs/design/<udtf_name>.md` describing the inputs, outputs, dependencies, error handling, and security considerations.
2. Add the implementation to `src/pyspark_udtf/udtfs/<udtf_name>.py`.
3. Export the public UDTF from `src/pyspark_udtf/udtfs/__init__.py`.
4. Add focused tests in `tests/test_<udtf_name>.py`.
5. Add a runnable example or README entry showing how to register and call the UDTF.

New UDTFs should have a clear, general-purpose use case. Keep network and service-specific logic separate from the Spark wrapper when practical so it can be tested without external access.

## Pull requests

Keep each pull request focused and explain:

- what problem it solves;
- how the implementation works;
- how it was tested;
- whether it changes public APIs or adds dependencies.

By contributing, you agree that your contribution is licensed under the [Apache License 2.0](LICENSE).
