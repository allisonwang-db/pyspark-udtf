import requests
import base64
from typing import Iterator, Tuple
from pyspark.sql.functions import udtf
from pyspark.sql.types import Row, StringType, StructType, StructField
from ..utils.version_check import require_pyspark_version

class BatchInferenceImageCaptionLogic:
    """
    A UDTF that generates image captions using batch inference.
    
    Args:
        batch_size (int): Number of images to process in a single batch.
        token (str): API token for authentication.
        endpoint (str): The model serving endpoint URL.
    """
    
    def __init__(self, batch_size: int, token: str, endpoint: str):
        self.batch_size = batch_size
        self.token = token
        self.endpoint = endpoint
        self.buffer = []

    def eval(self, url: str):
        """
        Processes each row. Buffers the image URL and triggers batch processing
        when the buffer size reaches the configured batch_size.
        """
        self.buffer.append(url)
        if len(self.buffer) >= self.batch_size:
            yield from self.process_batch()

    def terminate(self):
        """
        Called when all rows have been processed.
        Processes any remaining items in the buffer.
        """
        if self.buffer:
            yield from self.process_batch()

    @require_pyspark_version("4.1")
    def some_advanced_feature(self):
        """
        Placeholder for a feature that requires PySpark 4.1+.
        """
        pass

    def process_batch(self) -> Iterator[Row]:
        """
        sends a batch of images to the model serving endpoint.
        """
        if not self.buffer:
            return

        headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json"
        }
        
        try:
            # Simplified payload structure matching common serving endpoints
            payload = {
                "inputs": self.buffer
            }
            
            response = requests.post(
                self.endpoint,
                headers=headers,
                json=payload
            )
            response.raise_for_status()
            
            # Assuming response JSON contains a list of predictions/captions
            predictions = response.json().get('predictions', [])
            
            if len(predictions) != len(self.buffer):
                # Fallback if lengths mismatch
                for _ in self.buffer:
                    yield Row(caption="Error: API response mismatch")
            else:
                for caption in predictions:
                    yield Row(caption=str(caption))
                    
        except Exception as e:
            error_msg = f"Error processing batch: {str(e)}"
            for _ in self.buffer:
                yield Row(caption=error_msg)
        finally:
            self.buffer = []

# Register the UDTF
BatchInferenceImageCaption = udtf(
    BatchInferenceImageCaptionLogic, 
    returnType=StructType([StructField("caption", StringType())])
)
