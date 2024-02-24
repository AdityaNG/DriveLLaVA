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
    remove_noise,
)


def get_rotation_matrix(pose):
    """
    origin: (x, y, z, roll, pitch, yaw)
    """
    # Construct 3D rotation matrix
    origin = pose.copy()

    rot_x = np.array(
        [
            [1, 0, 0],
            [0, np.cos(origin[3]), -np.sin(origin[3])],
            [0, np.sin(origin[3]), np.cos(origin[3])],
        ]
    )
    rot_y = np.array(
        [
            [np.cos(origin[4]), 0, np.sin(origin[4])],
            [0, 1, 0],
            [-np.sin(origin[4]), 0, np.cos(origin[4])],
        ]
    )
    rot_z = np.array(
        [
            [np.cos(origin[5]), -np.sin(origin[5]), 0],
            [np.sin(origin[5]), np.cos(origin[5]), 0],
            [0, 0, 1],
        ]
    )

    rot = np.dot(rot_z, np.dot(rot_y, rot_x))

    return rot


def get_local_pose(
    pose: np.ndarray,
    frame_index: int,
    num_frames: int,
):
    """
    Gets the pose from a specific time index

    Rotates and translates the pose to the correct position
    """
    # pose: (1200, 6)
    assert pose.shape[0] == 1200
    assert pose.shape[0] - frame_index > num_frames

    sub_pose = pose[frame_index : frame_index + num_frames].copy()

    # origin = sub_pose[0]
    sub_pose[0, :] = 0

    # Each element is (x, y, z, roll, pitch, yaw)
    for i in range(1, sub_pose.shape[0]):
        rot = get_rotation_matrix(sub_pose[i])
        sub_pose[i, :3] += np.dot(sub_pose[:, :3], rot)[i - 1, :3]

    sub_pose[1:, :3] += sub_pose[:-1, :3]

    return sub_pose


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

        assert os.path.exists(pose_path), pose_path

        pose = np.load(pose_path)

        # Inches to meters
        pose[:, :3] *= 0.0254
        pose[:, 1] = 0.0
        # pose[:, 2] *= 0.01

        # pose[1:, :3] += pose[:-1, :3]

        # pose[:, :3] = smoothen_traj(pose[:, :3])
        # pose = smoothen_traj(pose, window_size=100)
        # pose = smoothen_traj(pose, window_size=20*3)
        pose = remove_noise(
            pose,
            window_length=21 * 2 -1,
            # window_length=9,
            polyorder=1,
        )

        pose[np.isnan(pose)] = 0
        pose[np.isinf(pose)] = 0

        print("pose", pose.shape, pose.dtype)

        for i in range(6):
            print(
                f"pose[:,{i}]",
                (np.min(pose[:, i]), np.max(pose[:, i])),
                np.mean(pose[:, i]),
                np.std(pose[:, i]),
            )

        # Iterate over the embeddings in batches and decode the images
        for i in tqdm(
            range(0, len(decoded_imgs_list) - NUM_FRAMES, 1), desc="Video"
        ):
            img = cv2.imread(decoded_imgs_list[i])
            sub_pose = get_local_pose(
                pose,
                i,
                NUM_FRAMES,
            )

            trajectory = sub_pose[:, :3]

            # Swap axes
            trajectory = trajectory[:, [2, 1, 0]]

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
