"""
Generates image frames for the commavq dataset
"""

import pickle

import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

from drivellava.constants import ENCODED_POSE_ALL
from drivellava.utils import (
    plot_bev_trajectory,
    plot_steering_traj,
)

from drivellava.trajectory_encoder import (
    NUM_TRAJECTORY_TEMPLATES,
    TRAJECTORY_SIZE,
)

from drivellava.datasets.commavq import CommaVQPoseDataset


def main():

    NUM_FRAMES = TRAJECTORY_SIZE
    WINDOW_LENGTH = 21 * 2 - 1
    SKIP_FRAMES = 20 * 1
    K = NUM_TRAJECTORY_TEMPLATES
    PLOT_TRAJ = False

    trajectories = []

    for pose_path in tqdm(
        ENCODED_POSE_ALL, desc="Loading trajectories"
    ):
        pose_dataset = CommaVQPoseDataset(
            pose_path,
            num_frames=NUM_FRAMES,
            window_length=WINDOW_LENGTH,
            polyorder=1,
        )

        # Iterate over the embeddings in batches and decode the images
        for i in tqdm(
            range(
                WINDOW_LENGTH,len(pose_dataset) - WINDOW_LENGTH, SKIP_FRAMES
            ),
            desc="Video",
            disable=True,
        ):
            img = np.zeros((128, 256, 3), dtype=np.uint8)

            trajectory = pose_dataset[i]

            trajectory_2d = trajectory[:, [0, 2]]

            trajectories.append(trajectory_2d.flatten())

            trajectory_2d_flipped = trajectory_2d.copy()
            trajectory_2d_flipped[:, 0] *= -1

            if PLOT_TRAJ:
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

    print('Running K-MEANs on the trajectories', len(trajectories))
    kmeans = KMeans(n_clusters=K, random_state=0).fit(trajectories)
    proposed_trajectory_templates = [centroid.reshape((NUM_FRAMES, 2)) for centroid in kmeans.cluster_centers_]

    # Save the kmeans object as trajectory_templates/kmeans.pkl
    with open(f'trajectory_templates/kmeans_{str(K)}.pkl', 'wb') as f:
        pickle.dump(kmeans, f)

    # Plotting the trajectory templates
    plt.figure(figsize=(10, 6))
    for template in proposed_trajectory_templates:
        plt.plot(template[:, 0], template[:, 1], alpha=0.5)
    plt.xlabel('X')
    plt.ylabel('Y')

    # X lim
    plt.xlim(-10, 10)
    # Y lim
    plt.ylim(-1, 40)

    plt.title('Proposed Trajectory Templates')
    plt.savefig(f'trajectory_templates/trajectory_templates_{str(K)}.png')  # Saving the plot
    plt.show()

    # Saving the templates as a NumPy file
    proposed_trajectory_templates_np = np.array(proposed_trajectory_templates, dtype=np.float32)
    print('proposed_trajectory_templates_np.shape', proposed_trajectory_templates_np.shape)
    np.save(f'trajectory_templates/proposed_trajectory_templates_{str(K)}.npy', proposed_trajectory_templates_np, allow_pickle=False)


if __name__ == "__main__":
    main()
