"""
Generates image frames for the commavq dataset
"""

import json
import os
import random
import sys

import cv2
import numpy as np
import onnxruntime as ort
from tqdm import tqdm

from drivellava.constants import (
    COMMAVQ_DIR,
    DECODER_ONNX_PATH,
    LLAVA_PATH,
    get_image_path,
    get_json,
)
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

if LLAVA_PATH not in sys.path:
    sys.path.append(LLAVA_PATH)

from llava.constants import DEFAULT_IMAGE_TOKEN  # noqa


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


def get_drivellava_prompt(trajectory_encoder: TrajectoryEncoder):
    traj_list = list(trajectory_encoder.token2trajectory.keys())
    random.shuffle(traj_list)
    traj_str = ",".join(list(map(str, traj_list)))
    P1 = (
        f"{DEFAULT_IMAGE_TOKEN}\nYou are DriveLLaVA, a "
        + "self-driving car. You will select the "
        + "appropriate trrajectory token given the "
        + "above image as context.\n"
        + "You may select one from the "
        + f"following templates: {traj_str}"
    )
    P2 = f"""{DEFAULT_IMAGE_TOKEN} As DriveLLaVA, the autonomous vehicle, your task is to analyze the given image and determine the optimal driving path. Choose the most suitable trajectory option from the list provided based on the visual information. {traj_str}"""  # noqa
    P3 = f"""{DEFAULT_IMAGE_TOKEN} You are the AI system DriveLLaVA, responsible for navigating self-driving cars. With the image provided as your guide, select the correct trajectory from the options below to ensure a safe and efficient route. {traj_str}"""  # noqa
    P4 = f"""{DEFAULT_IMAGE_TOKEN} Imagine yourself as DriveLLaVA, an advanced self-driving vehicle intelligence. Examine the scenario depicted in the image and decide on the best course of action by selecting an appropriate trajectory from the given templates. {traj_str}"""  # noqa
    P5 = f"""{DEFAULT_IMAGE_TOKEN} You embody DriveLLaVA, the brain behind autonomous driving technology. Given the context shown in the image, it's your job to pick the right trajectory from the available choices to navigate safely. {traj_str}"""  # noqa
    P6 = f"""{DEFAULT_IMAGE_TOKEN} As DriveLLaVA, a pioneering self-driving car AI, you're tasked with interpreting the visual cues in the provided image to choose the most suitable trajectory from the list of options to ensure a smooth journey. {traj_str}"""  # noqa
    P7 = f"""{DEFAULT_IMAGE_TOKEN} You, as DriveLLaVA, are at the forefront of autonomous navigation. Assess the situation depicted in the image and select the trajectory that best aligns with safe and efficient driving principles from the options provided. {traj_str}"""  # noqa
    P8 = f"""{DEFAULT_IMAGE_TOKEN} Functioning as DriveLLaVA, the self-driving car's decision-making system, you must look at the image and determine the best path forward by choosing from the predefined trajectory templates. {traj_str}"""  # noqa
    P9 = f"""{DEFAULT_IMAGE_TOKEN} You are DriveLLaVA, an AI designed for autonomous vehicles. Your objective is to analyze the context presented in the image and select a trajectory that guarantees the safety and comfort of your passengers from the given templates. {traj_str}"""  # noqa

    return random.choice([P1, P2, P3, P4, P5, P6, P7, P8, P9])


def generate_sparse_dataset(
    pose_path: str,
    pose_index: int,
    NUM_FRAMES: int,
    WINDOW_LENGTH: int,
    SKIP_FRAMES: int,
    trajectory_encoder: TrajectoryEncoder = None,  # type: ignore
    decoder_onnx: ort.InferenceSession = None,  # type: ignore
):
    batch_size = 1

    encoded_video_path = pose_path.replace("pose_data", "data").replace(
        "pose_val", "val"
    )

    json_path = get_json(encoded_video_path)

    # if os.path.isfile(json_path):
    #     return

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

        if not os.path.isfile(frame_path):
            embeddings_batch = embeddings[i : i + batch_size]

            if decoder_onnx is None:  # Lazy loading
                decoder_onnx = load_model_from_onnx_comma(
                    DECODER_ONNX_PATH, device="cuda"
                )
            frames = decode_image(
                decoder_onnx,
                embeddings_batch,
                batch_size,
            )
            frame = frames[0]
            os.makedirs(os.path.dirname(frame_path), exist_ok=True)
            cv2.imwrite(frame_path, frame)

        trajectory, trajectory_encoded = pose_dataset[i]

        rel_frame_path = frame_path.replace(COMMAVQ_DIR + "/", "")

        unique_id = pose_index * 100000 + i
        data += [
            {
                "id": str(unique_id),
                "image": rel_frame_path,
                "conversations": [
                    {
                        "from": "human",
                        "value": get_drivellava_prompt(trajectory_encoder),
                    },
                    {"from": "gpt", "value": trajectory_encoded},
                ],
            }
        ]

    # Write to json
    with open(json_path, "w", encoding="utf-8") as f:
        json_data = json.dumps(data, ensure_ascii=False, indent=4)
        f.write(json_data)


if __name__ == "__main__":
    print(get_drivellava_prompt(TrajectoryEncoder()))
