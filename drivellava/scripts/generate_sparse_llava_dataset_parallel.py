"""
Generates image frames for the commavq dataset using parallel processing
"""

import concurrent.futures

from tqdm import tqdm

from drivellava.constants import ENCODED_POSE_ALL
from drivellava.sparse_llava_dataset import generate_sparse_dataset
from drivellava.trajectory_encoder import TRAJECTORY_SIZE


def generate_frame(pose_path_num_frames_window_length_skip_frames):
    """
    Wrapper function to call generate_sparse_dataset with all necessary
    arguments.
    This is needed because ProcessPoolExecutor.map only supports functions
    with a single argument.
    """
    pose_path, num_frames, window_length, skip_frames = (
        pose_path_num_frames_window_length_skip_frames
    )
    generate_sparse_dataset(
        pose_path,
        num_frames,
        window_length,
        skip_frames,
    )


def main():

    NUM_FRAMES = TRAJECTORY_SIZE
    WINDOW_LENGTH = 21 * 2 - 1
    SKIP_FRAMES = 20 * 20

    # Prepare a list of arguments for each task
    tasks = [
        (pose_path, NUM_FRAMES, WINDOW_LENGTH, SKIP_FRAMES)
        for pose_path in ENCODED_POSE_ALL
    ]

    # Initialize progress bar
    pbar = tqdm(
        total=len(ENCODED_POSE_ALL), desc="Generating sparse LLaVA dataset"
    )

    # Use ProcessPoolExecutor to parallelize dataset generation
    with concurrent.futures.ProcessPoolExecutor() as executor:
        # Map the generate_frame function across all tasks
        # The result iterator allows us to update the progress bar
        # as tasks complete
        for _ in executor.map(generate_frame, tasks):
            pbar.update(1)

    pbar.close()


if __name__ == "__main__":
    main()
