import pytest
from unittest.mock import MagicMock, patch
import sys
from pyspark_udtf.utils.version_check import require_pyspark_version, check_version_compatibility

def test_require_pyspark_version_decorator():
    # Mocking pyspark version is handled in conftest, but we need to control it here
    # Since conftest sets it at module level, we might need to patch version.parse or pyspark.__version__
    
    # Check what conftest set: "4.0.0"
    
    @require_pyspark_version("3.5")
    def feature_old():
        return True
        
    @require_pyspark_version("4.0")
    def feature_current():
        return True
        
    @require_pyspark_version("4.1")
    def feature_future():
        return True

    # These should pass
    assert feature_old() is True
    assert feature_current() is True
    
    # This should fail
    with pytest.raises(ImportError) as excinfo:
        feature_future()
    assert "requires PySpark version >= 4.1" in str(excinfo.value)

def test_check_version_compatibility():
    assert check_version_compatibility("3.5") is True
    assert check_version_compatibility("4.0") is True
    assert check_version_compatibility("4.1") is False

