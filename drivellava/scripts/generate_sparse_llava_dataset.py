"""
Generates image frames for the commavq dataset
"""

from tqdm import tqdm

from drivellava.constants import ENCODED_POSE_ALL, DECODER_ONNX_PATH
from drivellava.sparse_llava_dataset import generate_sparse_dataset
from drivellava.onnx import load_model_from_onnx_comma
from drivellava.trajectory_encoder import TRAJECTORY_SIZE
from drivellava.trajectory_encoder import (
    NUM_TRAJECTORY_TEMPLATES,
    TRAJECTORY_SIZE,
    TRAJECTORY_TEMPLATES_KMEANS_PKL,
    TRAJECTORY_TEMPLATES_NPY,
    TrajectoryEncoder,
)

def main():

    NUM_FRAMES = TRAJECTORY_SIZE
    WINDOW_LENGTH = 21 * 2 - 1
    SKIP_FRAMES = 20 * 20

    decoder_onnx = load_model_from_onnx_comma(
        DECODER_ONNX_PATH, device="cuda"
    )

    trajectory_encoder = TrajectoryEncoder(
        num_trajectory_templates=NUM_TRAJECTORY_TEMPLATES,
        trajectory_size=TRAJECTORY_SIZE,
        trajectory_templates_npy=TRAJECTORY_TEMPLATES_NPY,
        trajectory_templates_kmeans_pkl=TRAJECTORY_TEMPLATES_KMEANS_PKL,
    )

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
            trajectory_encoder=trajectory_encoder,
            decoder_onnx=decoder_onnx,
        )


if __name__ == "__main__":
    main()
