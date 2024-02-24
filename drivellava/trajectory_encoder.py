import os
import pickle

import numpy as np

NUM_TRAJECTORY_TEMPLATES = 128
TRAJECTORY_SIZE = 20
TRAJECTORY_TEMPLATES_NPY = f"./trajectory_templates/proposed_trajectory_templates_{NUM_TRAJECTORY_TEMPLATES}.npy"  # noqa
TRAJECTORY_TEMPLATES_KMEANS_PKL = (
    f"./trajectory_templates/kmeans_{NUM_TRAJECTORY_TEMPLATES}.pkl"
)


class TrajectoryEncoder:

    def __init__(
        self,
        num_trajectory_templates=NUM_TRAJECTORY_TEMPLATES,
        trajectory_size=TRAJECTORY_SIZE,
        trajectory_templates_npy=TRAJECTORY_TEMPLATES_NPY,
        trajectory_templates_kmeans_pkl=TRAJECTORY_TEMPLATES_KMEANS_PKL,
    ) -> None:
        self.num_trajectory_templates = num_trajectory_templates
        self.trajectory_templates_npy = trajectory_templates_npy
        self.trajectory_templates_kmeans_pkl = trajectory_templates_kmeans_pkl
        self.trajectory_size = trajectory_size

        assert os.path.exists(
            trajectory_templates_npy
        ), f"File {trajectory_templates_npy} does not exist"
        assert os.path.exists(
            trajectory_templates_kmeans_pkl
        ), f"File {trajectory_templates_kmeans_pkl} does not exist"

        self.num_trajectory_templates = num_trajectory_templates
        self.trajectory_templates = np.load(
            trajectory_templates_npy, allow_pickle=False
        )
        # Validate
        assert self.trajectory_templates.shape == (
            self.num_trajectory_templates,
            self.trajectory_size,
            2,
        ), (
            f"Expected trajectory_templates.shape to be "
            f"({self.num_trajectory_templates}, {self.trajectory_size}, 2), "
            f"got {self.trajectory_templates.shape}"
        )

        with open(trajectory_templates_kmeans_pkl, "rb") as f:
            self.kmeans = pickle.load(f)

        self.start_index = 0

        self.token2trajectory = {
            chr(i): self.trajectory_templates[i - self.start_index]
            for i in range(
                self.start_index,
                self.start_index + self.num_trajectory_templates,
            )
        }
        self.trajectory_index_2_token = {
            i - self.start_index: chr(i)
            for i in range(
                self.start_index,
                self.start_index + self.num_trajectory_templates,
            )
        }

        self.TOKEN_IDS = [
            chr(i) for i in range(self.start_index, 2 * self.start_index)
        ]

    def encode(self, trajectory_3d) -> str:
        N, _ = trajectory_3d.shape
        assert N == self.trajectory_size, (
            f"Expected trajectory_3d.shape[0] to be "
            f"{self.trajectory_size}, got {N}"
        )
        # trajectory is torch.Tensor of shape (TRAJECTORY_SIZE, 3)
        trajectory_2d = trajectory_3d[:, [0, 2]]
        trajectory_2d = trajectory_2d.astype(float)
        # trajectory is torch.Tensor of shape (TRAJECTORY_SIZE, 2)
        trajectory_index = self.kmeans.predict(
            trajectory_2d.reshape((1, self.trajectory_size * 2))
        )[0]

        token = self.trajectory_index_2_token[trajectory_index]

        # Ensure token is a string of length 1
        assert isinstance(token, str) and len(token) == 1, token

        return token

    def decode(self, token: str):
        assert isinstance(token, str)
        trajectory_2d = self.token2trajectory[token]

        height_axis = np.zeros_like(trajectory_2d[:, 0])
        trajectory_2d = np.stack(
            (
                trajectory_2d[:, 0],
                height_axis,
                trajectory_2d[:, 1],
            ),
            axis=-1,
        )

        # trajectory_templates is of shape (B, F, 100, 2)
        trajectory_2d = trajectory_2d.reshape((self.trajectory_size, 3))
        trajectory_2d = trajectory_2d.astype(np.float32)
        return trajectory_2d
