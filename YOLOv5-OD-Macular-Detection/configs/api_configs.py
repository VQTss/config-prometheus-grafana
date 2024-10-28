import numpy as np
import io
import cv2
import torch

# ================================================================= #

def  get_image_from_url(image_content: bytes) -> np.ndarray:
    image_stream = io.BytesIO(image_content)
    image = cv2.imdecode(np.frombuffer(image_stream.getvalue(), np.uint8), 1)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return image

def get_model_YOLOv5():
    model = torch.hub.load("./yolov5", "custom", "./yolov5/runs/train/exp9/weights/last.pt", source="local")
    # if torch.cuda.device_count() > 1:
    #     model = torch.nn.DataParallel(model)

    return model
