import io
import requests
import numpy as np
import json
import cv2
import base64
from time import time
from fastapi import FastAPI, File, Query
from fastapi.security import HTTPBasic
import uvicorn
import logging

from configs.api_configs import get_image_from_url, get_model_YOLOv5

# Setup logging to display in the terminal
logger = logging.getLogger("yolov5_app")
logger.setLevel(logging.DEBUG)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# Prometheus monitoring libraries
from prometheus_client import start_http_server
from opentelemetry import metrics
from opentelemetry.exporter.prometheus import PrometheusMetricReader
from opentelemetry.metrics import set_meter_provider
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.resources import SERVICE_NAME, Resource

# Start Prometheus client
start_http_server(port=8099, addr='0.0.0.0')

# Metrics setup
resource = Resource(attributes={SERVICE_NAME: "yolov5"})
reader = PrometheusMetricReader()
provider = MeterProvider(resource=resource, metric_readers=[reader])
set_meter_provider(provider)

# Define metrics with simplified names
meter = metrics.get_meter("detection", "0.1.2")

# Total request counter
yolov5_request_counter = meter.create_counter(
    name="yolov5_request_total",
    description="Total number of YOLOv5 requests",
)

# Successful requests counter
yolov5_request_success_counter = meter.create_counter(
    name="yolov5_request_success_total",
    description="Count of successful YOLOv5 requests",
)

# Error requests counter
yolov5_request_error_counter = meter.create_counter(
    name="yolov5_request_error_total",
    description="Count of failed YOLOv5 requests",
)

# Response time histogram
yolov5_response_latency = meter.create_histogram(
    name="yolov5_response_latency_seconds",
    description="YOLOv5 response latency in seconds",
    unit="seconds"
)

model = get_model_YOLOv5()
app = FastAPI()
security = HTTPBasic()

@app.post("/od-macular-detection")
async def od_macular_detection(image_url: str = Query(..., description="URL of the image to process.")):
    start_time = time()
    label = {"api": "/od-macular-detection"}

    try:
        response = requests.get(image_url)
        response.raise_for_status()
        mime_type = response.headers.get('Content-Type', 'image/png')
        input_image = get_image_from_url(response.content)
        
        outputs = model(input_image)
        outputs.render()
        
        # Convert and encode image to base64
        for im in outputs.ims:
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            _, encoded_image = cv2.imencode('.png', im)
            base64_image = base64.b64encode(encoded_image).decode("utf-8")

        # Convert output bounding boxes to JSON format
        od_macular = outputs.pandas().xyxy[0].to_json(orient="records")
        od_macular = json.loads(od_macular)
        results = {
            "annotated_image": f"data:{mime_type};base64,{base64_image}",
            "od_macular": od_macular
        }

        # Record successful request count
        logger.debug(f"Incrementing success counter for {label}")
        yolov5_request_success_counter.add(1, label)
        
    except requests.exceptions.RequestException as e:
        # Record error request count and log the error
        logger.error(f"Failed to retrieve image: {e}")
        yolov5_request_error_counter.add(1, label)
        return {"error": f"Failed to retrieve image from URL: {e}"}

    end_time = time()
    elapsed_time = end_time - start_time
    logger.debug(f"Recording response time for {label}: {elapsed_time}")
    
    # Record response time and total request count
    yolov5_response_latency.record(elapsed_time, label)
    yolov5_request_counter.add(1, label)

    # Return the results
    return results

if __name__ == '__main__':
    app_logger = logging.getLogger("uvicorn.error")
    app_logger.setLevel(logging.DEBUG)
    uvicorn.run(app=app, host='0.0.0.0', port=7005, log_level="debug")
