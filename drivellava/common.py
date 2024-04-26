import base64
from textwrap import dedent

import cv2
import numpy as np

KMPH_2_MPS = 1 / 3.6
DEG_2_RAD = np.pi / 180

DEFAULT_VISION_MODEL = "gpt-4-vision-preview"
DEFAULT_CONTROLS_MODEL = "gpt-3.5-turbo"
DEFAULT_LLM_PROVIDER = "openai"
# DEFAULT_LLM_PROVIDER = "ollama"

DEFAULT_MISSION = dedent(
    """As DriveLLaVA, the autonomous vehicle, your task is to analyze the \
    given image and determine the optimal driving path. Choose the most \
    suitable trajectory option from the list provided based on the \
    visual information. Make sure to stay centered in your lane. \
    If you deviate from the lane make sure to make course corrections"""
)


def encode_opencv_image(img):
    _, buffer = cv2.imencode(".jpg", img)
    jpg_as_text = base64.b64encode(buffer).decode("utf-8")
    return jpg_as_text


def encode_opencv_image_buf(img, compression_factor=95):
    _, buffer = cv2.imencode(
        ".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), compression_factor]
    )
    return buffer


def decode_opencv_image_buf(buf):
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)
