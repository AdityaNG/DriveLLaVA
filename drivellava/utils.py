import cv2
import numpy as np
import torch
import torchvision
from scipy.signal import savgol_filter

comma_inv_transform = torchvision.transforms.Compose(
    [
        # Convert to uint8
        torchvision.transforms.Lambda(
            lambda x: (((x + 1.0) * 127.5).clamp(0, 255)).byte()
        ),
        # Permute to (H, W, C)
        torchvision.transforms.Lambda(lambda x: x.permute(1, 2, 0)),
        # Squueze to remove the batch dimension
        torchvision.transforms.Lambda(lambda x: x.squeeze(0)),
        # To numpy
        torchvision.transforms.Lambda(lambda x: x.detach().cpu().numpy()),
        torchvision.transforms.Lambda(
            lambda x: cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
        ),
    ]
)


def comma_inv_transform_batch(
    frame_y_pred,
    B,
    device,
):
    future_frames = []
    for batch_index in range(B):
        frame_y_pred_transformed = comma_inv_transform(
            frame_y_pred[batch_index]
        )
        future_frames.append(frame_y_pred_transformed)
    future_frames = torch.tensor(future_frames, device=device)
    return future_frames


def convert_3D_points_to_2D(points_3D, homo_cam_mat):
    points_2D = []
    for index in range(points_3D.shape[0]):
        p4d = points_3D[index]
        p2d = (homo_cam_mat) @ p4d
        px, py = 0, 0
        if p2d[2][0] != 0.0:
            px, py = int(p2d[0][0] / p2d[2][0]), int(p2d[1][0] / p2d[2][0])

        points_2D.append([px, py])

    return np.array(points_2D)


def get_rect_coords(x_i, y_i, x_j, y_j, width=2.83972):
    Pi = np.array([x_i, y_i])
    Pj = np.array([x_j, y_j])
    height = np.linalg.norm(Pi - Pj)
    diagonal = (width**2 + height**2) ** 0.5
    D = diagonal / 2.0

    M = ((Pi + Pj) / 2.0).reshape((2,))
    theta = np.arctan2(Pi[1] - Pj[1], Pi[0] - Pj[0])
    theta += np.pi / 4.0
    points = np.array(
        [
            M
            + np.array(
                [
                    D * np.sin(theta + 0 * np.pi / 2.0),
                    D * np.cos(theta + 0 * np.pi / 2.0),
                ]
            ),
            M
            + np.array(
                [
                    D * np.sin(theta + 1 * np.pi / 2.0),
                    D * np.cos(theta + 1 * np.pi / 2.0),
                ]
            ),
            M
            + np.array(
                [
                    D * np.sin(theta + 2 * np.pi / 2.0),
                    D * np.cos(theta + 2 * np.pi / 2.0),
                ]
            ),
            M
            + np.array(
                [
                    D * np.sin(theta + 3 * np.pi / 2.0),
                    D * np.cos(theta + 3 * np.pi / 2.0),
                ]
            ),
        ]
    )
    return points


def get_rect_coords_3D(Pi, Pj, width=0.25):
    x_i, y_i = Pi[0, 0], Pi[2, 0]
    x_j, y_j = Pj[0, 0], Pj[2, 0]
    points_2D = get_rect_coords(x_i, y_i, x_j, y_j, width)
    points_3D = []
    for index in range(points_2D.shape[0]):
        # point_2D = points_2D[index]
        point_3D = Pi.copy()
        point_3D[0, 0] = points_2D[index, 0]
        point_3D[2, 0] = points_2D[index, 1]

        points_3D.append(point_3D)

    return np.array(points_3D)


def plot_steering_traj(
    frame_center,
    trajectory,
    color=(255, 0, 0),
    intrinsic_matrix=None,
    DistCoef=None,
    # offsets=[0.0, 1.5, 1.0],
    offsets=[0.0, -0.75, 0.0],
    # offsets=[0.0, -1.5, 0.0],
    method="add_weighted",
    track=True,
):
    assert method in ("overlay", "mask", "add_weighted")

    h, w = frame_center.shape[:2]

    if intrinsic_matrix is None:
        # intrinsic_matrix = np.array([
        #     [525.5030,         0,    333.4724],
        #     [0,         531.1660,    297.5747],
        #     [0,              0,    1.0],
        # ])
        intrinsic_matrix = np.array(
            [
                [525.5030, 0, w / 2],
                [0, 531.1660, h / 2],
                [0, 0, 1.0],
            ]
        )
    if DistCoef is None:
        DistCoef = np.array(
            [
                0.0177,
                3.8938e-04,  # Tangential Distortion
                -0.1533,
                0.4539,
                -0.6398,  # Radial Distortion
            ]
        )
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(
        intrinsic_matrix, DistCoef, (w, h), 1, (w, h)
    )
    homo_cam_mat = np.hstack((intrinsic_matrix, np.zeros((3, 1))))

    # rot = trajectory[0][:3,:3]
    # rot = np.eye(3,3)
    prev_point = None
    prev_point_3D = None
    rect_frame = np.zeros_like(frame_center)

    for trajectory_point in trajectory:
        p4d = np.ones((4, 1))
        p3d = np.array(
            [
                trajectory_point[0] * 1 - offsets[0],
                # trajectory_point[1] * 1 - offsets[1],
                -offsets[1],
                trajectory_point[2] * 1 - offsets[2],
            ]
        ).reshape((3, 1))
        # p3d = np.linalg.inv(rot) @ p3d
        p4d[:3, :] = p3d

        p2d = (homo_cam_mat) @ p4d
        if (
            p2d[2][0] != 0.0
            and not np.isnan(p2d).any()
            and not np.isinf(p2d).any()
        ):
            px, py = int(p2d[0][0] / p2d[2][0]), int(p2d[1][0] / p2d[2][0])
            # frame_center = cv2.circle(frame_center, (px, py), 2, color, -1)
            if prev_point is not None:
                px_p, py_p = prev_point
                dist = ((px_p - px) ** 2 + (py_p - py) ** 2) ** 0.5
                if dist < 20:
                    if track:
                        rect_coords_3D = get_rect_coords_3D(p4d, prev_point_3D)
                        rect_coords = convert_3D_points_to_2D(
                            rect_coords_3D, homo_cam_mat
                        )
                        rect_frame = cv2.fillPoly(
                            rect_frame, pts=[rect_coords], color=color
                        )

                    frame_center = cv2.line(
                        frame_center, (px_p, py_p), (px, py), color, 2
                    )
                    # break

            prev_point = (px, py)
            prev_point_3D = p4d.copy()
        else:
            prev_point = None
            prev_point_3D = None

    if method == "mask":
        mask = np.logical_and(
            rect_frame[:, :, 0] == color[0],
            rect_frame[:, :, 1] == color[1],
            rect_frame[:, :, 2] == color[2],
        )
        frame_center[mask] = color
    elif method == "overlay":
        frame_center += (0.2 * rect_frame).astype(np.uint8)
    elif method == "add_weighted":
        cv2.addWeighted(frame_center, 1.0, rect_frame, 0.2, 0.0, frame_center)
    return frame_center


def plot_bev_trajectory(trajectory, frame_center, color=(0, 255, 0)):
    WIDTH, HEIGHT = frame_center.shape[1], frame_center.shape[0]
    traj_plot = np.ones((HEIGHT, WIDTH, 3), dtype=np.uint8) * 255

    Z = trajectory[:, 2]
    X = trajectory[:, 0]

    RAN = 20.0
    X_min, X_max = -RAN, RAN
    # Z_min, Z_max = -RAN, RAN
    Z_min, Z_max = -0.1 * RAN, RAN
    X = (X - X_min) / (X_max - X_min)
    Z = (Z - Z_min) / (Z_max - Z_min)

    # X = (X - lb) / (ub - lb)
    # Z = (Z - lb) / (ub - lb)

    for traj_index in range(1, X.shape[0]):
        u = int(round(np.clip((X[traj_index] * (WIDTH - 1)), -1, WIDTH + 1)))
        v = int(round(np.clip((Z[traj_index] * (HEIGHT - 1)), -1, HEIGHT + 1)))
        u_p = int(
            round(np.clip((X[traj_index - 1] * (WIDTH - 1)), -1, WIDTH + 1))
        )
        v_p = int(
            round(np.clip((Z[traj_index - 1] * (HEIGHT - 1)), -1, HEIGHT + 1))
        )

        if u < 0 or u >= WIDTH or v < 0 or v >= HEIGHT:
            continue

        traj_plot = cv2.circle(traj_plot, (u, v), 2, color, -1)
        traj_plot = cv2.line(traj_plot, (u_p, v_p), (u, v), color, 2)

    traj_plot = cv2.flip(traj_plot, 0)
    return traj_plot


def smoothen_traj(trajectory, window_size=4):
    """
    Smoothen a trajectory using moving average.

    Args:
    trajectory (list): List of 3D points [(x1, y1, z1), (x2, y2, z2), ...].
        window_size (int): Size of the moving average window.

    Returns:
        list: Smoothened trajectory as a list of 3D points.
    """
    smoothed_traj = []
    num_points = len(trajectory)

    half_window = window_size // 2

    # Handle edge cases
    if num_points <= window_size:
        return trajectory

    # Calculate the moving average for each point
    for i in range(num_points):
        window_start = max(0, i - half_window + 1)
        window_end = min(i + half_window + 1, num_points)
        window_points = trajectory[window_start:window_end]
        # avg_point = (
        #     sum(p[0] for p in window_points) / (window_end - window_start),
        #     sum(p[1] for p in window_points) / (window_end - window_start),
        #     sum(p[2] for p in window_points) / (window_end - window_start),
        #     sum(p[3] for p in window_points) / (window_end - window_start),
        #     sum(p[4] for p in window_points) / (window_end - window_start),
        #     sum(p[5] for p in window_points) / (window_end - window_start),
        # )
        avg_point = np.mean(window_points, axis=0)

        smoothed_traj.append(avg_point)

    smoothed_traj = np.array(smoothed_traj)

    return smoothed_traj


def remove_noise(pose_matrix, window_length=5, polyorder=2):
    """
    Applies a Savitzky-Golay filter to remove high-frequency noise from the
    pose data.

    Parameters:
    - pose_matrix (numpy.ndarray): An (N, 6) array representing the motion of
      a car, with each pose being (x, y, z, roll, pitch, yaw) in meters
      and radians.
    - window_length (int): The length of the filter window (i.e., the number
      of coefficients).
      `window_length` must be a positive odd integer.
    - polyorder (int): The order of the polynomial used to fit the samples.
      `polyorder` must be less than `window_length`.

    Returns:
    - numpy.ndarray: The smoothed pose matrix.
    """
    if window_length % 2 == 0:
        raise ValueError("window_length must be an odd integer")
    if polyorder >= window_length:
        raise ValueError("polyorder must be less than window_length")

    smoothed_pose = np.zeros_like(pose_matrix)
    for i in range(pose_matrix.shape[1]):
        smoothed_pose[:, i] = savgol_filter(
            pose_matrix[:, i], window_length, polyorder
        )

    return smoothed_pose


@torch.no_grad()
def decode_image(
    decoder_onnx,
    embeddings,
    batch_size,
    device=torch.device("cpu"),
):
    encoding_indices_future_onnx = embeddings.astype(np.int64)
    frames = decoder_onnx.run(
        None, {"encoding_indices": encoding_indices_future_onnx}
    )[0]

    frames = torch.tensor(frames).to(device=device)

    frames = comma_inv_transform_batch(
        frames,
        batch_size,
        device,
    )

    frames = frames.cpu().numpy()

    return frames
