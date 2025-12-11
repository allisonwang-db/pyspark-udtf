import requests
import base64
from typing import Iterator, Tuple
from pyspark.sql.functions import udtf
from pyspark.sql.types import Row, StringType, StructType, StructField
from ..utils.version_check import check_version_compatibility

check_version_compatibility("3.5")

class BatchInferenceImageCaptioLogic:
    def __init__(self):
        self.batch_size = 3
        self.vision_endpoint = "databricks-claude-3-7-sonnet"
        self.workspace_url = "<workspace-url>"
        self.image_buffer = []
        self.results = []
        
    def eval(self, row, api_token):
        self.image_buffer.append((str(row[0]), api_token))
        if len(self.image_buffer) >= self.batch_size:
            self._process_batch()
    
    def terminate(self):
        if self.image_buffer:
            self._process_batch()
        for caption in self.results:
            yield (caption,)
    
    def _process_batch(self):
        batch_data = self.image_buffer.copy()
        self.image_buffer.clear()
        
        import base64
        import httpx
        import requests

        # API request timeout in seconds
        api_timeout = 60
        # Maximum tokens for vision model response
        max_response_tokens = 300
        # Temperature controls randomness (lower = more deterministic)
        model_temperature = 0.3
        
        # create a batch for the images
        batch_images = []
        api_token = batch_data[0][1] if batch_data else None
        
        for image_url, _ in batch_data:
            image_response = httpx.get(image_url, timeout=15)
            image_data = base64.standard_b64encode(image_response.content).decode("utf-8")
            batch_images.append(image_data)
        
        content_items = [{
            "type": "text",
            "text": "Provide brief captions for these images, one per line."
        }]
        for img_data in batch_images:
            content_items.append({
                "type": "image_url",
                "image_url": {
                    "url": "data:image/jpeg;base64," + img_data
                }
            })
        
        payload = {
            "messages": [{
                "role": "user",
                "content": content_items
            }],
            "max_tokens": max_response_tokens,
            "temperature": model_temperature
        }
        
        response = requests.post(
            self.workspace_url + "/serving-endpoints/" +
            self.vision_endpoint + "/invocations",
            headers={
                'Authorization': 'Bearer ' + api_token,
                'Content-Type': 'application/json'
            },
            json=payload,
            timeout=api_timeout
        )
        
        result = response.json()
        batch_response = result['choices'][0]['message']['content'].strip()
        
        lines = batch_response.split('\n')
        captions = [line.strip() for line in lines if line.strip()]
        
        while len(captions) < len(batch_data):
            captions.append(batch_response)
        
        self.results.extend(captions[:len(batch_data)])

BatchInferenceImageCaption = udtf(
    BatchInferenceImageCaptionLogic, 
    returnType=StructType([StructField("caption", StringType())])
)
