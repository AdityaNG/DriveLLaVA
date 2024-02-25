"""
Generates image frames for the commavq dataset
"""

from tqdm import tqdm

from drivellava.constants import ENCODED_POSE_ALL
from drivellava.sparse_llava_dataset import generate_sparse_dataset
from drivellava.trajectory_encoder import TRAJECTORY_SIZE


def main():

    NUM_FRAMES = TRAJECTORY_SIZE
    WINDOW_LENGTH = 21 * 2 - 1
    SKIP_FRAMES = 20 * 20

    for pose_index, pose_path in tqdm(
        enumerate(ENCODED_POSE_ALL),
        desc="Generating sparse LLaVA dataset",
        total=len(ENCODED_POSE_ALL),
    ):

        generate_sparse_dataset(
            pose_path,
            pose_index,
            NUM_FRAMES,
            WINDOW_LENGTH,
            SKIP_FRAMES,
        )


if __name__ == "__main__":
    main()
