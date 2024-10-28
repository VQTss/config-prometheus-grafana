from time import sleep
import requests
from loguru import logger

def predict():
    logger.info("Sending POST request to OD Macular Detection!")
    
    # The image URL parameter for the API
    image_url = "https://upload.wikimedia.org/wikipedia/commons/2/26/Fundus_of_eye_normal.jpg"
    
    # Construct the full URL with the image_url parameter
    url = f"http://192.168.1.12:7005/od-macular-detection?image_url={image_url}"
    
    # Sending the POST request with the required headers
    response = requests.post(
        url,
        headers={
            "accept": "application/json",
        },
        data='',  # No additional data is sent in the body
    )
    
    # Log the response status code and content
    logger.info(f"Response Status Code: {response.status_code}")
    # logger.info(f"Response Content: {response.text}")


if __name__ == "__main__":
    while True:
        predict()
        sleep(1)
