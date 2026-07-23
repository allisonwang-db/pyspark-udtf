# Python UDTF Examples for PySpark

[![PyPI](https://img.shields.io/pypi/v/pyspark-udtf.svg)](https://pypi.org/project/pyspark-udtf/)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

Practical, tested Python UDTF (User-Defined Table Function) examples for Apache Spark and PySpark 4. Learn how to write, register, test, package, and run Python UDTFs for fuzzy matching, batch inference, reverse ETL, and Databricks Unity Catalog.

## What is a Python UDTF?

A Python UDTF transforms each input row into zero or more output rows. Unlike a scalar Python UDF, which returns one value per input row, a UDTF returns a table and can be called from Spark SQL with a `TABLE` argument. This repository provides reusable implementations and focused examples for common PySpark data-processing patterns.

## Included examples

| Example | Use case | Source |
| --- | --- | --- |
| Fuzzy matching | Find the closest candidate for a string and return its similarity score | [`fuzzy_match.py`](src/pyspark_udtf/udtfs/fuzzy_match.py) |
| Batch image captioning | Buffer image URLs and send efficient batched requests to a model endpoint | [`image_caption.py`](src/pyspark_udtf/udtfs/image_caption.py) |
| Meta Conversions API | Send conversion events from Spark as a reverse ETL workload | [`meta_capi.py`](src/pyspark_udtf/udtfs/meta_capi.py) |
| Unity Catalog registration | Convert a PySpark UDTF into a governed Unity Catalog function | [Unity Catalog guide](docs/unity_catalog_udtf.md) |

## Installation

You can quickly install the package using pip:

```bash
pip install pyspark-udtf
```

## Usage

### Fuzzy Matching (Quick Start)

This UDTF demonstrates how to use Python's standard library `difflib` to perform fuzzy string matching in PySpark. It takes a target string and a list of candidates, returning the best match and a similarity score.

```python
from pyspark.sql import SparkSession
from pyspark_udtf.udtfs import FuzzyMatch

spark = SparkSession.builder.getOrCreate()

# Register the UDTF
spark.udtf.register("fuzzy_match", FuzzyMatch)

# Create a sample dataframe with typos
data = [
    ("aple", ["apple", "banana", "orange"]),
    ("bananna", ["apple", "banana", "orange"]),
    ("orange", ["apple", "banana", "orange"]),
    ("grape", ["apple", "banana", "orange"])
]
df = spark.createDataFrame(data, ["typo", "candidates"])

# Use the UDTF in SQL
df.createOrReplaceTempView("typos")

spark.sql("""
    SELECT *
    FROM fuzzy_match(TABLE(SELECT typo, candidates FROM typos))
""").show()
```

### Batch Inference Image Captioning

This UDTF demonstrates how to perform efficient batch inference against a model serving endpoint. It buffers rows and sends them in batches to reduce network overhead.

```python
from pyspark.sql import SparkSession
from pyspark_udtf.udtfs import BatchInferenceImageCaption

spark = SparkSession.builder.getOrCreate()

# Register the UDTF
spark.udtf.register("batch_image_caption", BatchInferenceImageCaption)

# View UDTF definition and parameters
help(BatchInferenceImageCaption.func)

# Usage in SQL
# Assuming you have a table 'images' with a column 'url'
spark.sql("""
    SELECT *
    FROM batch_image_caption(
        TABLE(SELECT url FROM images),
        10,  -- batch_size
        'your-api-token',
        'https://your-endpoint.com/score'
    )
""").show()
```

## Requirements

- Python >= 3.10
- PySpark >= 4.0.0
- requests
- pandas
- pyarrow

## Documentation

For more detailed Python UDTF documentation, including design documents and guides for Unity Catalog integration, see the [`docs/`](docs/) directory.

- [Unity Catalog Guide](docs/unity_catalog_udtf.md)
- [Meta Conversions API design](docs/design/meta_capi.md)

## Contributing

Contributions are welcome. Read the [contributing guide](CONTRIBUTING.md) for development setup, testing requirements, and the process for proposing a new Python UDTF.

For help using the project, see [support resources](SUPPORT.md). To report a vulnerability, follow the [security policy](SECURITY.md).

## Development

We recommend using [uv](https://github.com/astral-sh/uv) for extremely fast package management.

```bash
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install the package
uv add pyspark-udtf
```

### Running Tests

To run the test suite:

```bash
# Run all tests
uv run pytest

# Run specific test file
uv run pytest tests/test_image_caption.py
```

### Linting

This project uses [Ruff](https://docs.astral.sh/ruff/) for linting and formatting. Install dev dependencies, then run:

```bash
uv sync --extra dev   # install ruff
uv run ruff check .   # lint
uv run ruff format .  # format
```

### Adding Dependencies

To add a new runtime dependency:

```bash
uv add package_name
```

To add a development dependency:

```bash
uv add --dev package_name
```

### Bumping Version

You can bump the version automatically using `uv` (requires uv >= 0.7.0):

```bash
# Bump patch version (0.1.0 -> 0.1.1)
uv version --bump patch

# Bump minor version (0.1.0 -> 0.2.0)
uv version --bump minor
```

Alternatively, you can manually update `pyproject.toml`:

1. Open `pyproject.toml`.
2. Update the `version` field under `[project]`:
   ```toml
   [project]
   version = "0.1.1"  # Update this value
   ```

### Publishing to PyPI

To build and publish the package to PyPI:

1. **Build the package:**
   ```bash
   uv build
   ```
   This will create distributions in the `dist/` directory.

2. **Publish to PyPI:**
   ```bash
   uv publish
   ```
   Note: You will need to configure your PyPI credentials (API token) either via environment variables (`UV_PUBLISH_TOKEN`) or following `uv`'s authentication documentation.

## Codex Skills

This repository includes Codex skills for common development tasks in [`.agents/skills/`](.agents/skills/). Invoke a skill by mentioning its `$skill-name` in your Codex prompt.

### create-udtf

Use this skill when you want to **create, write, or generate a new PySpark UDTF**. It guides you through:

1. **Analyze requirements** – Determine inputs, outputs, and external dependencies
2. **Design** – Create a design doc in `docs/design/<udtf_name>.md` (required for all UDTFs)
3. **Implementation** – Implement the UDTF in `src/pyspark_udtf/udtfs/<udtf_name>.py`
4. **Registration** – Add the UDTF to `src/pyspark_udtf/udtfs/__init__.py`
5. **Testing** – Add tests in `tests/test_<udtf_name>.py`

**When to use:** Ask Codex to create a new UDTF or explicitly invoke `$create-udtf` when describing the UDTF you want to build.

**Reference implementations:**

- Simple UDTF: `src/pyspark_udtf/udtfs/fuzzy_match.py`
- Complex UDTF (buffering, external API): `src/pyspark_udtf/udtfs/meta_capi.py`

### release-package

Explicitly invoke `$release-package` to bump the package version, build the distributions, publish the confirmed artifacts to PyPI, and push the release commit. The skill requires confirmation before publishing or force-pushing.
