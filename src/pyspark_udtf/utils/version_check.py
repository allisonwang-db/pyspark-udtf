import pyspark
from packaging import version
import warnings
import functools

def get_pyspark_version():
    return version.parse(pyspark.__version__)

def require_pyspark_version(min_version: str):
    """
    Decorator to check if the installed PySpark version meets the requirement.
    Raises an ImportError if the version is insufficient.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            current = get_pyspark_version()
            required = version.parse(min_version)
            if current < required:
                raise ImportError(
                    f"This feature requires PySpark version >= {min_version}. "
                    f"Current version is {current}."
                )
            return func(*args, **kwargs)
        return wrapper
    return decorator

def check_version_compatibility(min_version: str):
    """
    Checks version compatibility and returns True/False.
    Useful for conditional logic inside functions.
    """
    current = get_pyspark_version()
    required = version.parse(min_version)
    return current >= required

