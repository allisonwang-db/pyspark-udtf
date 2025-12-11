# PySpark UDTF Examples

A collection of Python User-Defined Table Functions (UDTFs) for PySpark, demonstrating how to leverage UDTFs for complex data processing tasks.

## Requirements

- Python >= 3.10
- PySpark >= 4.0.0
- requests
- pandas
- pyarrow

## Installation

```bash
pip install pyspark-udtf
```

## Usage

### Batch Inference Image Captioning

This UDTF demonstrates how to perform efficient batch inference against a model serving endpoint. It buffers rows and sends them in batches to reduce network overhead.

```python
from pyspark.sql import SparkSession
from pyspark_udtf.udtfs import BatchInferenceImageCaption

spark = SparkSession.builder.getOrCreate()

# Register the UDTF
spark.udtf.register("batch_image_caption", BatchInferenceImageCaption)

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

# Usage in Python API
from pyspark.sql.functions import lit

images_df = spark.createDataFrame([
    ("http://example.com/image1.jpg",),
    ("http://example.com/image2.jpg",)
], ["url"])

# Use the UDTF class directly
output_df = BatchInferenceImageCaption(
    lit(10), 
    lit("your-api-token"), 
    lit("https://your-endpoint.com/score")
).func(images_df.select("url"))
# Note: Syntax for direct class usage might vary depending on PySpark 4.0 API specifics for UDTFs.
# The standard way is via SQL or registering it.
```

## Available UDTFs

### `BatchInferenceImageCaption`

Performs batch inference for image captioning.

**Arguments:**
- `batch_size` (int): Number of images to process in a single batch.
- `token` (str): API token for authentication.
- `endpoint` (str): The model serving endpoint URL.

**Input:**
- A table with image URLs (or other input data expected by your model).

**Output:**
- A struct containing the `caption` (string).

## Development

1. Install dependencies:
   ```bash
   pip install -e ".[dev]"
   ```

2. Run tests:
   ```bash
   pytest
   ```

