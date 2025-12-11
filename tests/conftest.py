import sys
from unittest.mock import MagicMock

# Create a mock for pyspark
pyspark_mock = MagicMock()
pyspark_mock.__version__ = "4.0.0"

# Mock types
class Row:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
    def __eq__(self, other):
        return self.__dict__ == other.__dict__
    def __repr__(self):
        return f"Row({self.__dict__})"

types_mock = MagicMock()
types_mock.Row = Row
types_mock.StringType = MagicMock()
types_mock.StructType = MagicMock()
types_mock.StructField = MagicMock()

# Mock functions
functions_mock = MagicMock()
# udtf decorator mock: returns the class (or a wrapper around it)
def udtf_mock(*args, **kwargs):
    def decorator(cls):
        return cls
    return decorator

functions_mock.udtf = udtf_mock

# Mock sql
sql_mock = MagicMock()
sql_mock.types = types_mock
sql_mock.functions = functions_mock

pyspark_mock.sql = sql_mock

# Apply mocks to sys.modules
sys.modules["pyspark"] = pyspark_mock
sys.modules["pyspark.sql"] = sql_mock
sys.modules["pyspark.sql.types"] = types_mock
sys.modules["pyspark.sql.functions"] = functions_mock

