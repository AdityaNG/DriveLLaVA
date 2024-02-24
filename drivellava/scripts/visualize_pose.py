"""
Generates image frames for the commavq dataset
"""

import os

import cv2
import numpy as np
from tqdm import tqdm

from drivellava.constants import DECODED_IMGS_ALL_AVAILABLE
from drivellava.utils import (
    plot_bev_trajectory,
    plot_steering_traj,
)

from drivellava.datasets.commavq import CommaVQPoseDataset


def main():

    NUM_FRAMES = 20 * 1

    for encoded_video_path in tqdm(
        DECODED_IMGS_ALL_AVAILABLE.keys(), desc="npy files"
    ):
        decoded_imgs_list = DECODED_IMGS_ALL_AVAILABLE[encoded_video_path]
        pose_path = encoded_video_path.replace("data_", "pose_data_").replace(
            "val", "pose_val"
        )
        print(pose_path)

        pose_dataset = CommaVQPoseDataset(
            pose_path,
            num_frames=NUM_FRAMES,
            window_length=21 * 2 - 1,
            polyorder=1,
        )

        # Iterate over the embeddings in batches and decode the images
        for i in tqdm(
            range(0, len(decoded_imgs_list) - NUM_FRAMES, 1), desc="Video"
        ):
            img = cv2.imread(decoded_imgs_list[i])

            trajectory = pose_dataset[i]

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
            )

            img_bev = plot_bev_trajectory(trajectory, img)

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

            cv2.imshow(
                "frame_path_bev", cv2.resize(img_bev, (0, 0), fx=2, fy=2)
            )

            key = cv2.waitKey(1)

            if key == ord("q"):
                exit()
            elif key == ord("n"):
                break


if __name__ == "__main__":
    main()
