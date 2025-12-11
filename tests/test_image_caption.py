import pytest
from unittest.mock import Mock, patch
from pyspark.sql.types import Row
from pyspark_udtf.udtfs.image_caption import BatchInferenceImageCaptionLogic

@pytest.fixture
def udtf_instance():
    return BatchInferenceImageCaptionLogic(
        batch_size=2,
        token="fake-token",
        endpoint="http://fake-endpoint"
    )

def test_eval_buffering(udtf_instance):
    # Test that eval buffers and doesn't yield until batch size is met
    
    # First item - should be buffered
    results1 = list(udtf_instance.eval("http://image1.jpg"))
    assert len(results1) == 0
    assert len(udtf_instance.buffer) == 1
    
    # Second item - should trigger processing
    with patch('requests.post') as mock_post:
        mock_response = Mock()
        mock_response.json.return_value = {'predictions': ['caption1', 'caption2']}
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response
        
        results2 = list(udtf_instance.eval("http://image2.jpg"))
        
        assert len(results2) == 2
        assert results2[0].caption == 'caption1'
        assert results2[1].caption == 'caption2'
        assert len(udtf_instance.buffer) == 0 # Buffer should be cleared
        
        mock_post.assert_called_once()
        assert mock_post.call_args[1]['json']['inputs'] == ["http://image1.jpg", "http://image2.jpg"]

def test_terminate_flushes_buffer(udtf_instance):
    # Test that terminate processes remaining items
    
    # Add one item (less than batch size of 2)
    list(udtf_instance.eval("http://image1.jpg"))
    assert len(udtf_instance.buffer) == 1
    
    with patch('requests.post') as mock_post:
        mock_response = Mock()
        mock_response.json.return_value = {'predictions': ['caption1']}
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response
        
        results = list(udtf_instance.terminate())
        
        assert len(results) == 1
        assert results[0].caption == 'caption1'
        assert len(udtf_instance.buffer) == 0
        
        mock_post.assert_called_once()

def test_error_handling(udtf_instance):
    # Test that API errors are handled gracefully
    
    with patch('requests.post') as mock_post:
        mock_post.side_effect = Exception("API Error")
        
        # Trigger batch processing immediately by setting batch_size=1 for this check or just filling it
        udtf_instance.buffer = ["http://image1.jpg", "http://image2.jpg"]
        
        results = list(udtf_instance.process_batch())
        
        assert len(results) == 2
        assert "Error processing batch" in results[0].caption
        assert "Error processing batch" in results[1].caption
        assert len(udtf_instance.buffer) == 0 # Buffer should still be cleared

