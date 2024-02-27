import os

import numpy as np

from drivellava.trajectory_encoder import TrajectoryEncoder
from drivellava.utils import remove_noise, smoothen_traj


class CommaVQPoseDataset:
    def __init__(
        self,
        pose_path: str,
        num_frames: int = 20,
        window_length: int = 21 * 2 - 1,
        polyorder: int = 1,
    ):
        assert os.path.exists(pose_path), pose_path
        self.pose_path = pose_path
        self.num_frames = num_frames
        pose = np.load(pose_path)

        # Inches to meters
        pose[:, :3] *= 0.0254
        pose[:, 1] = 0.0

        try:
            pose = remove_noise(
                pose,
                window_length=window_length,
                polyorder=polyorder,
            )
        except Exception as ex:
            print("Warning: ", ex)
            pose = smoothen_traj(
                pose,
                window_size=window_length,
            )

        pose[np.isnan(pose)] = 0
        pose[np.isinf(pose)] = 0

        self.pose = pose

    def __len__(self):
        return len(self.pose)

    def __getitem__(self, index):
        sub_pose = get_local_pose(
            self.pose,
            index,
            self.num_frames,
        )

        trajectory = sub_pose[:, :3].copy()

        # Swap axes
        trajectory = trajectory[:, [2, 1, 0]]

        return trajectory


class CommaVQPoseQuantizedDataset(CommaVQPoseDataset):

    def __init__(self, *args, **kwargs):
        assert "trajectory_encoder" in kwargs
        trajectory_encoder = kwargs.pop("trajectory_encoder")
        assert isinstance(trajectory_encoder, TrajectoryEncoder)

        super().__init__(*args, **kwargs)

        self.trajectory_encoder = trajectory_encoder

    def __getitem__(self, index):
        trajectory = super().__getitem__(index)
        trajectory_encoded = self.trajectory_encoder.encode(trajectory)
        return trajectory, trajectory_encoded


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
