"""
Evaluated DriveLLaVA on a video sequence
"""

import os

import cv2
import numpy as np
from tqdm import tqdm

from drivellava.constants import get_image_path
from drivellava.datasets.commavq import CommaVQPoseQuantizedDataset
from drivellava.model import DriveLLaVA
from drivellava.sparse_llava_dataset import get_drivellava_prompt
from drivellava.trajectory_encoder import (
    NUM_TRAJECTORY_TEMPLATES,
    TRAJECTORY_SIZE,
    TRAJECTORY_TEMPLATES_KMEANS_PKL,
    TRAJECTORY_TEMPLATES_NPY,
    TrajectoryEncoder,
)
from drivellava.utils import plot_bev_trajectory, plot_steering_traj

# import sys


def main():

    # sys.path.append(LLAVA_PATH)

    # from transformers.models.llava.configuration_llava import LlavaConfig

    # fine_tuned_model_path = "liuhaotian/llava-v1.5-7b"
    fine_tuned_model_path = os.path.expanduser(
        "~/Datasets/checkpoints/checkpoint-600/"
    )

    args = type(
        "Args",
        (),
        {
            "model_path": fine_tuned_model_path,
            "model_base": None,
            # "model_name": get_model_name_from_path(fine_tuned_model_path),
            # "query": prompt,
            "conv_mode": "llava_llama_2",
            # "image_file": image_file,
            # "sep": ",",
            "temperature": 0,
            "top_p": None,
            "num_beams": 1,
            "max_new_tokens": 64,
        },
    )()

    trajectory_encoder = TrajectoryEncoder(
        num_trajectory_templates=NUM_TRAJECTORY_TEMPLATES,
        trajectory_size=TRAJECTORY_SIZE,
        trajectory_templates_npy=TRAJECTORY_TEMPLATES_NPY,
        trajectory_templates_kmeans_pkl=TRAJECTORY_TEMPLATES_KMEANS_PKL,
    )
    model = DriveLLaVA(args, trajectory_encoder)

    print(dir(model.tokenizer))
    # print(model.tokenizer.get_vocab())

    NUM_FRAMES = 20 * 1

    # encoded_video_path = "/root/Datasets/commavq/val/fe809f0fff5562cc4d2bdc073d242123_31.npy"  # noqa
    encoded_video_path = "/root/Datasets/commavq/data_0_to_2500/000e83c564317de4668c2cb372f89b91_6.npy"  # noqa
    # encoded_video_path = "/root/Datasets/commavq/img_data_0_to_2500/000e83c564317de4668c2cb372f89b91_6.npy"  # noqa

    # assert os.path.isfile(encoded_video_path), encoded_video_path

    pose_path = encoded_video_path.replace("data_", "pose_data_").replace(
        "val",
        "pose_val",
    )
    assert os.path.isfile(pose_path), pose_path

    decoded_imgs_list = []

    for frame_index in range(1200):
        frame_path = get_image_path(encoded_video_path, frame_index)
        frame_path = frame_path.replace("data_", "img_data_")
        # print('frame_path', frame_path)
        if os.path.isfile(frame_path):
            decoded_imgs_list.append(frame_path)

    pose_dataset = CommaVQPoseQuantizedDataset(
        pose_path,
        num_frames=NUM_FRAMES,
        window_length=21 * 2 - 1,
        polyorder=1,
        trajectory_encoder=trajectory_encoder,
    )

    # Save to video
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = None

    # Iterate over the embeddings in batches and decode the images
    for i in tqdm(
        range(0, len(decoded_imgs_list) - NUM_FRAMES, 1),
        desc="Video",
    ):
        if not os.path.isfile(decoded_imgs_list[i]):
            continue
        img = cv2.imread(decoded_imgs_list[i])

        trajectory, trajectory_encoded = pose_dataset[i]
        trajectory_quantized = trajectory_encoder.decode(trajectory_encoded)

        traj_tokens = model.tokenizer.tokenize(trajectory_encoded)
        traj_tokens_encoded = model.tokenizer.encode(trajectory_encoded)
        print(
            "traj_tokens",
            trajectory_encoded,
            "->",
            traj_tokens,
            "->",
            traj_tokens_encoded,
        )

        model_trajectory_quantized = model.run(
            get_drivellava_prompt(trajectory_encoder, default_image_token=""),
            [
                decoded_imgs_list[i],
            ],
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

        img = plot_steering_traj(
            img,
            model_trajectory_quantized,
            color=(0, 0, 255),
        )

        img_bev_gt = plot_bev_trajectory(trajectory, img, color=(255, 0, 0))
        img_bev_gtq = plot_bev_trajectory(
            trajectory_quantized, img, color=(0, 255, 0)
        )
        img_bev_pred = plot_bev_trajectory(
            model_trajectory_quantized, img, color=(0, 0, 255)
        )

        # Overlay BEVs
        img_bev = cv2.addWeighted(img_bev_gt, 0.5, img_bev_gtq, 0.5, 0)
        img_bev = cv2.addWeighted(img_bev, 0.5, img_bev_pred, 0.5, 0)

        # Write speed on img
        font = cv2.FONT_HERSHEY_SIMPLEX
        bottomLeftCornerOfText = (10, 50)
        fontScale = 0.5
        fontColor = (255, 255, 255)
        lineType = 2

        img = cv2.resize(img, (0, 0), fx=2, fy=2)
        img_bev = cv2.resize(img_bev, (0, 0), fx=2, fy=2)

        cv2.putText(
            img,
            f"Speed: {speed_mph.mean():.2f} mph",
            bottomLeftCornerOfText,
            font,
            fontScale,
            fontColor,
            lineType,
        )

        vis = np.concatenate([img, img_bev], axis=1)

        if out is None:
            out = cv2.VideoWriter(
                "test_media/trajectory.mp4", fourcc, 20.0, vis.shape[1::-1]
            )

        out.write(vis)
        cv2.imwrite("test_media/vis.png", vis)

    out.release()


if __name__ == "__main__":
    main()
