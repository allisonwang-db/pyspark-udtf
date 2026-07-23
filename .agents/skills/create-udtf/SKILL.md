---
name: create-udtf
description: Design, implement, register, and test a new PySpark User-Defined Table Function (UDTF). Use when the user wants to create, write, or generate a Python UDTF for this repository; do not use for Unity Catalog CREATE FUNCTION registration.
---

# Create PySpark UDTF

## Workflow

### 1. Analyze requirements

Determine the inputs, outputs, behavior, and external dependencies from the user's request.

### 2. Design

Create `docs/design/<udtf_name>.md` for every new UDTF. Include:

- An overview of the UDTF.
- A copy-pastable Python example that defines every DataFrame, schema, mapping, and other variable it uses.
- Placeholder values such as `YOUR_API_KEY` for secrets.
- The input arguments and output schema.
- The expected buffering, state-management, and external-API behavior.

Keep the design focused on the interface and observable behavior rather than full implementation details. Present the design and get the user's approval before implementing it.

### 3. Implement

Create `src/pyspark_udtf/udtfs/<udtf_name>.py`.

- Use `src/pyspark_udtf/udtfs/fuzzy_match.py` for a simple reference or `src/pyspark_udtf/udtfs/meta_capi.py` for buffering and external API patterns.
- Use type hints and `yield` rows from the UDTF.
- Prefer a static `returnType` in the `@udtf` decorator. Implement `analyze` only when the output schema cannot be determined statically.
- Handle `TABLE` arguments efficiently and use `terminate` to flush buffers or clean up resources.
- Keep the implementation in one file unless logic is shared or unusually complex.
- Copy the complete usage example from the design document into the class docstring.

Do not use this skill to create Unity Catalog Python UDTFs with `CREATE FUNCTION` syntax.

### 4. Register

Export the new UDTF from `src/pyspark_udtf/udtfs/__init__.py`.

### 5. Test

Create `tests/test_<udtf_name>.py`.

- Follow `tests/test_image_caption.py` or `tests/test_meta_capi.py` for established patterns.
- Test `eval` directly and test `analyze` when implemented.
- Mock external dependencies such as `requests`.
- Verify the output schema and returned data.
- Run the focused test, `uv run pytest`, and `uv run ruff check .`.

## Project context

Read and follow the repository root `AGENTS.md` before making changes. Prefer `TABLE` arguments for whole-table or partition processing.

For API details, consult the [PySpark UDTF documentation](https://spark.apache.org/docs/latest/api/python/tutorial/sql/python_udtf.html) and [Databricks UDTF documentation](https://docs.databricks.com/aws/en/udf/python-udtf).
