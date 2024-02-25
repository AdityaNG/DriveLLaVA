"""
Generates image frames for the commavq dataset
"""

import json
import os
import onnxruntime as ort

import cv2
import numpy as np
from tqdm import tqdm

from drivellava.constants import DECODER_ONNX_PATH, get_image_path, get_json
from drivellava.datasets.commavq import CommaVQPoseQuantizedDataset
from drivellava.onnx import load_model_from_onnx_comma
from drivellava.trajectory_encoder import (
    NUM_TRAJECTORY_TEMPLATES,
    TRAJECTORY_SIZE,
    TRAJECTORY_TEMPLATES_KMEANS_PKL,
    TRAJECTORY_TEMPLATES_NPY,
    TrajectoryEncoder,
)
from drivellava.utils import (
    decode_image,
    plot_bev_trajectory,
    plot_steering_traj,
)


def visualize_pose(
    frame_path: str,
    trajectory_encoder: TrajectoryEncoder,
    trajectory_encoded: str,
    trajectory: np.ndarray,
):
    img = cv2.imread(frame_path)

    trajectory_quantized = trajectory_encoder.decode(trajectory_encoded)

    print(
        "trajectory[0]",
        (np.min(trajectory[:, 0]), np.max(trajectory[:, 0])),
    )
    print(
        "trajectory[1]",
        (np.min(trajectory[:, 1]), np.max(trajectory[:, 1])),
    )
    print(
        "trajectory[2]",
        (np.min(trajectory[:, 2]), np.max(trajectory[:, 2])),
    )
    dx = trajectory[1:, 2] - trajectory[:-1, 2]
    speed = dx / (1.0 / 20.0)
    # m/s to km/h
    speed_kmph = speed * 3.6
    # speed mph
    speed_mph = speed_kmph * 0.621371

    img = plot_steering_traj(
        img,
        trajectory,
        color=(255, 0, 0),
    )

    img = plot_steering_traj(
        img,
        trajectory_quantized,
        color=(0, 255, 0),
    )

    img_bev = plot_bev_trajectory(trajectory, img, color=(255, 0, 0))
    img_bev = plot_bev_trajectory(trajectory_quantized, img, color=(0, 255, 0))

    # Write speed on img
    font = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (10, 50)
    fontScale = 0.5
    fontColor = (255, 255, 255)
    lineType = 2

    img = cv2.resize(img, (0, 0), fx=2, fy=2)

    cv2.putText(
        img,
        f"Speed: {speed_mph.mean():.2f} mph",
        bottomLeftCornerOfText,
        font,
        fontScale,
        fontColor,
        lineType,
    )

    cv2.imshow("frame_path", img)

    cv2.imshow("frame_path_bev", cv2.resize(img_bev, (0, 0), fx=2, fy=2))

    key = cv2.waitKey(1)

    if key == ord("q"):
        exit()


def generate_sparse_dataset(
    pose_path: str,
    pose_index: int,
    NUM_FRAMES: int,
    WINDOW_LENGTH: int,
    SKIP_FRAMES: int,
    trajectory_encoder: TrajectoryEncoder = None,
    decoder_onnx: ort.InferenceSession = None,
):
    batch_size = 1
    if decoder_onnx is None:
        decoder_onnx = load_model_from_onnx_comma(DECODER_ONNX_PATH, device="cuda")

    encoded_video_path = pose_path.replace("pose_data", "data").replace(
        "pose_val", "val"
    )

    json_path = get_json(encoded_video_path)

    if os.path.isfile(json_path):
        return

    if trajectory_encoder is None:
        trajectory_encoder = TrajectoryEncoder(
            num_trajectory_templates=NUM_TRAJECTORY_TEMPLATES,
            trajectory_size=TRAJECTORY_SIZE,
            trajectory_templates_npy=TRAJECTORY_TEMPLATES_NPY,
            trajectory_templates_kmeans_pkl=TRAJECTORY_TEMPLATES_KMEANS_PKL,
        )

    pose_dataset = CommaVQPoseQuantizedDataset(
        pose_path,
        num_frames=NUM_FRAMES,
        window_length=WINDOW_LENGTH,
        polyorder=1,
        trajectory_encoder=trajectory_encoder,
    )

    data = []

    assert os.path.exists(encoded_video_path)

    # embeddings: (1200, 8, 16) -> (B, x, y)
    embeddings = np.load(encoded_video_path)

    # Iterate over the embeddings in batches and decode the images
    for i in tqdm(
        range(WINDOW_LENGTH, len(pose_dataset) - WINDOW_LENGTH, SKIP_FRAMES),
        desc="Video",
        disable=True,
    ):
        frame_path = get_image_path(encoded_video_path, i)

        trajectory, trajectory_encoded = pose_dataset[i]

        if not os.path.isfile(frame_path):
            embeddings_batch = embeddings[i : i + batch_size]
            frames = decode_image(
                decoder_onnx,
                embeddings_batch,
                batch_size,
            )
            frame = frames[0]
            os.makedirs(os.path.dirname(frame_path), exist_ok=True)
            cv2.imwrite(frame_path, frame)

        unique_id = pose_index * 100000 + i
        data += [
            {
                "id": str(unique_id),
                "image": frame_path,
                "conversations": [
                    {
                        "from": "human",
                        "value": (
                            "<image>\nYou are DriveLLaVA, a "
                            + "self-driving car. You will select the "
                            + "appropriate trrajectory token given the "
                            + "above image as context.\n"
                            + "You may select one from the "
                            + "following templates: "
                            + ",".join(
                                trajectory_encoder.token2trajectory.keys()
                            )
                        ),
                    },
                    {"from": "gpt", "value": trajectory_encoded},
                ],
            }
        ]

    # Write to json
    with open(json_path, "w") as f:
        json.dump(data, f, indent=4)
