"""
Generates image frames for the commavq dataset
"""

import os

import cv2
import numpy as np
from tqdm import tqdm

from drivellava.constants import (
    DECODER_ONNX_PATH,
    ENCODED_VIDEOS_ALL,
    get_image_path,
)
from drivellava.onnx import load_model_from_onnx_comma
from drivellava.utils import decode_image


def main():

    batch_size = 4

    decoder_onnx = load_model_from_onnx_comma(DECODER_ONNX_PATH, device="cuda")

    for encoded_video_path in tqdm(ENCODED_VIDEOS_ALL, desc="npy files"):

        skip = True

        for frame_index in range(1200):
            frame_path = get_image_path(encoded_video_path, frame_index)
            if not os.path.exists(frame_path):
                skip = False
                break

        if skip:
            print(f"Skipping {encoded_video_path}")
            continue

        # embeddings: (1200, 8, 16) -> (B, x, y)
        embeddings = np.load(encoded_video_path)

        assert embeddings.shape[0] == 1200

        # Iterate over the embeddings in batches and decode the images
        for i in tqdm(range(0, len(embeddings), batch_size), desc="Video"):
            # Skip check
            if all(
                os.path.exists(get_image_path(encoded_video_path, x))
                for x in range(i, i + batch_size)
            ):
                print(f"Skipping {encoded_video_path} {i}:{i + batch_size}")
                continue

            embeddings_batch = embeddings[i : i + batch_size]
            frames = decode_image(
                decoder_onnx,
                embeddings_batch,
                batch_size,
            )

            # Save the frames
            for j, frame in enumerate(frames):
                frame_index = i + j
                frame_path = get_image_path(encoded_video_path, frame_index)
                os.makedirs(os.path.dirname(frame_path), exist_ok=True)
                # frame = (frame *).astype(np.uint8)
                cv2.imwrite(frame_path, frame)
                cv2.imshow("frame_path", cv2.resize(frame, (0, 0), fx=2, fy=2))

                cv2.waitKey(1)


if __name__ == "__main__":
    main()
