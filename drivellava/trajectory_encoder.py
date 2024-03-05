import json
import os
import pickle
from typing import List

import numpy as np

from drivellava.constants import VOCAB_JSON

NUM_TRAJECTORY_TEMPLATES = 16
TRAJECTORY_SIZE = 20
TRAJECTORY_TEMPLATES_NPY = f"./trajectory_templates/proposed_trajectory_templates_{NUM_TRAJECTORY_TEMPLATES}.npy"  # noqa
TRAJECTORY_TEMPLATES_KMEANS_PKL = (
    f"./trajectory_templates/kmeans_{NUM_TRAJECTORY_TEMPLATES}.pkl"
)
ENCODING = "utf-8"


class TrajectoryEncoder:

    def __init__(
        self,
        num_trajectory_templates=NUM_TRAJECTORY_TEMPLATES,
        trajectory_size=TRAJECTORY_SIZE,
        trajectory_templates_npy=TRAJECTORY_TEMPLATES_NPY,
        trajectory_templates_kmeans_pkl=TRAJECTORY_TEMPLATES_KMEANS_PKL,
        vocab_json=VOCAB_JSON,
    ) -> None:
        self.num_trajectory_templates = num_trajectory_templates
        self.trajectory_templates_npy = trajectory_templates_npy
        self.trajectory_templates_kmeans_pkl = trajectory_templates_kmeans_pkl
        self.trajectory_size = trajectory_size

        with open(vocab_json, "r", encoding=ENCODING) as f:
            self.vocab_json = json.load(f)

        self.vocab_json_inv = {v: k for k, v in self.vocab_json.items()}

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

        self.start_token_id = 31500
        self.end_token_id = self.start_token_id + self.num_trajectory_templates

        self.TOKEN_IDS: List[str] = [
            self.vocab_json_inv[i + self.start_token_id]
            for i in range(self.num_trajectory_templates)
        ]

        index = 0

        while len(self.TOKEN_IDS) < self.num_trajectory_templates:

            token = chr(index)

            if len(repr(token.encode(ENCODING).decode(ENCODING))) == 3:
                self.TOKEN_IDS.append(str(token))

            index += 1

        self.token2trajectory = {
            tok: self.trajectory_templates[i]
            for i, tok in enumerate(self.TOKEN_IDS)
        }
        self.trajectory_index_2_token = {
            i: tok for i, tok in enumerate(self.TOKEN_IDS)
        }

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
        trajectory_3d = np.stack(
            (
                trajectory_2d[:, 0],
                height_axis,
                trajectory_2d[:, 1],
            ),
            axis=-1,
        )

        # trajectory_templates is of shape (B, F, 100, 2)
        trajectory_3d = trajectory_3d.reshape((self.trajectory_size, 3))
        trajectory_3d = trajectory_3d.astype(np.float32)
        return trajectory_3d

    def left_to_right(
        self,
    ):
        """Arrange the tokens from left to right
        -ve x is left
        +ve x is right
        """
        x = self.trajectory_templates[:, :, 0]
        # x_mean, x_std = x.mean(), x.std()
        # y_mean, y_std = (
        #     self.trajectory_templates[:, :, 1].mean(),
        #     self.trajectory_templates[:, :, 1].std(),
        # )

        # x_mean is ~0
        # Sort the trajectory_templates by the mean of the x axis
        sorted_indices = np.argsort(x.mean(axis=1))

        # Split the sorted_indices into left, center and right
        left = sorted_indices[: self.num_trajectory_templates // 3]
        center = sorted_indices[
            self.num_trajectory_templates
            // 3 : 2
            * self.num_trajectory_templates
            // 3
        ]
        right = sorted_indices[2 * self.num_trajectory_templates // 3 :]

        left_tokens = [self.trajectory_index_2_token[i] for i in left]
        center_tokens = [self.trajectory_index_2_token[i] for i in center]
        right_tokens = [self.trajectory_index_2_token[i] for i in right]

        # print("self.trajectory_templates", self.trajectory_templates.shape)
        # print("left", left.shape)
        # print("center", center.shape)
        # print("right", right.shape)

        # print("left_tokens", left_tokens)
        # print("center_tokens", center_tokens)
        # print("right_tokens", right_tokens)

        return left_tokens, center_tokens, right_tokens


if __name__ == "__main__":
    trajectory_encoder = TrajectoryEncoder()

    trajectory_encoder.left_to_right()

    print(
        [
            i.encode(ENCODING).decode(ENCODING)
            for i in trajectory_encoder.TOKEN_IDS
        ]
    )
