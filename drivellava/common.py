import base64
from textwrap import dedent

import cv2
import numpy as np

KMPH_2_MPS = 1 / 3.6
DEG_2_RAD = np.pi / 180

DEFAULT_LLM_PROVIDER = "openai"
DEFAULT_VISION_MODEL = "gpt-4-vision-preview"
DEFAULT_CONTROLS_MODEL = "gpt-3.5-turbo"

# DEFAULT_LLM_PROVIDER = "ollama"
# DEFAULT_VISION_MODEL = "llava"
# DEFAULT_CONTROLS_MODEL = "llama3"

DEFAULT_MISSION = dedent("""Explore the town while obeying traffic laws""")


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
