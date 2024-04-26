"""
Generates image frames for the commavq dataset
"""

import pickle

import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from tqdm import tqdm

from drivellava.trajectory_encoder import TRAJECTORY_SIZE
from drivellava.utils import plot_bev_trajectory, plot_steering_traj


def generate_trajectory_dataset(
    num_templates: int,
    num_frames: int,
    total_angle: float = np.pi,  # radians
    trajectory_distance: float = 40.0,  # meters
    noise: float = 0.0,  # percentage of noise
) -> list:
    """
    Generate a list of trajectories.
    The list must be of length num_templates
    Each trajectory is of the following shape: (num_frames, 2)
    Each trajectory element consists of a 2D point (x, y)

    The list of trajectories must be generated such that the trajectories
    sweep in terms of curvature from left to right.
    i.e. the 0th trajectiry curves to the left,
        the (num_templates//2)th trajectory is straight, and
        the last trajectiry curves to the right.

    The maximum radius of curvature is max_radius_of_curvature.
    The distance traversed along any trajectory must be trajectory_distance.
    All the points on the trajectory must be roughly equidistant from their
        neighbors.
    """

    assert noise >= 0
    trajectory_dataset = []

    # Calculate the total distance and angle covered by each trajectory
    total_distance = trajectory_distance
    distance_per_frame = total_distance / num_frames

    for i in range(num_templates):
        # Calculate the starting point and orientation of the current
        # trajectory
        start_point = (0, 0, 0)

        # Initialize the current trajectory with the starting point
        trajectory = [start_point]

        traj_total_angle = total_angle / (i + 1)
        angle_step = traj_total_angle / num_frames

        for j in range(1, num_frames):
            # Calculate the new angle and position based on the previous one
            angle = (j - 1) * angle_step
            # angle = (num_frames - j -1) * angle_step
            # angle = np.pi - angle

            # Calculate the new position based on the previous one and the
            # curvature
            x = trajectory[-1][0] + distance_per_frame * np.sin(angle)
            y = trajectory[-1][2] + distance_per_frame * np.cos(angle)

            x = x * (1 + np.random.uniform(-noise, noise))
            y = y * (1 + np.random.uniform(-noise, noise))

            # Append the new point to the current trajectory
            trajectory.append((x, 0, y))
            # trajectory.append((y, 0, -x))

        # Add the current trajectory to the dataset
        trajectory_dataset.append(np.array(trajectory))
        print(
            "x",
            trajectory_dataset[-1][:, 0].min(),
            trajectory_dataset[-1][:, 0].max(),
        )
        print(
            "y",
            trajectory_dataset[-1][:, 2].min(),
            trajectory_dataset[-1][:, 2].max(),
        )

    return trajectory_dataset


def main():

    NUM_FRAMES = TRAJECTORY_SIZE
    K = 5
    PLOT_TRAJ = True

    trajectories = []

    trajectory_dataset = generate_trajectory_dataset(K * 10, NUM_FRAMES)

    # Iterate over the embeddings in batches and decode the images
    for i in tqdm(
        range(len(trajectory_dataset)),
        desc="Video",
        disable=True,
    ):

        trajectory = trajectory_dataset[i]

        trajectory_2d = trajectory[:, [0, 2]]

        trajectories.append(trajectory_2d.flatten())

        trajectory_2d_flipped = trajectory_2d.copy()
        trajectory_2d_flipped[:, 0] *= -1

        trajectories.append(trajectory_2d_flipped.flatten())

        if PLOT_TRAJ:
            img = np.zeros((128, 256, 3), dtype=np.uint8)

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

    print("Running K-MEANs on the trajectories", len(trajectories))
    kmeans = KMeans(n_clusters=K, random_state=0).fit(trajectories)
    proposed_trajectory_templates = [
        centroid.reshape((NUM_FRAMES, 2))
        for centroid in kmeans.cluster_centers_
    ]

    # Save the kmeans object as trajectory_templates/kmeans.pkl
    with open(f"trajectory_templates/kmeans_simple_{str(K)}.pkl", "wb") as f:
        pickle.dump(kmeans, f)

    # Plotting the trajectory templates
    plt.figure(figsize=(10, 6))
    for template in proposed_trajectory_templates:
        plt.plot(template[:, 0], template[:, 1], alpha=0.5)
    plt.xlabel("X")
    plt.ylabel("Y")

    # X lim
    plt.xlim(-10, 10)
    # Y lim
    plt.ylim(-1, 40)

    plt.title("Proposed Trajectory Templates")
    plt.savefig(
        f"trajectory_templates/trajectory_templates_simple_{str(K)}.png"
    )  # Saving the plot
    plt.show()

    # Saving the templates as a NumPy file
    proposed_trajectory_templates_np = np.array(
        proposed_trajectory_templates, dtype=np.float32
    )
    print(
        "proposed_trajectory_templates_np.shape",
        proposed_trajectory_templates_np.shape,
    )
    np.save(
        f"trajectory_templates/proposed_trajectory_templates_simple_{str(K)}.npy",  # noqa
        proposed_trajectory_templates_np,
        allow_pickle=False,
    )


if __name__ == "__main__":
    main()
