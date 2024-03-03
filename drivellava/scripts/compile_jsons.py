"""
Trains LLAVA model on the cumulative dataset.
"""

import json
import os
import random
from typing import List

from drivellava.constants import ENCODED_JSON, VAL_ENCODED_JSON
from drivellava.sparse_llava_dataset import get_drivellava_prompt
from drivellava.trajectory_encoder import (
    TrajectoryEncoder, NUM_TRAJECTORY_TEMPLATES
)

def load_json_dataset(
    json_list: List[str],
    trajectory_encoder: TrajectoryEncoder,
):

    data = []
    for json_path in json_list:
        with open(json_path, "r", encoding="utf-8") as f:
            loaded = json.load(f)
            for index in range(len(loaded)):
                assert len(loaded[index]["conversations"][1]["value"]) == 1

                loaded[index]["conversations"][1]["value"] = (
                    "Selected Trajectory: "
                    + loaded[index]["conversations"][1]["value"]
                )
                loaded[index]["conversations"][0]["value"] = (
                    get_drivellava_prompt(trajectory_encoder)
                )
            data.extend(loaded)

    return data


def main():

    trajectory_encoder = TrajectoryEncoder()

    train = load_json_dataset(
        ENCODED_JSON,
        trajectory_encoder,
    )
    val = load_json_dataset(
        VAL_ENCODED_JSON,
        trajectory_encoder,
    )

    print(f"Train: {len(train)}")
    print(f"Val: {len(val)}")

    # Shuffle train and val
    random.shuffle(train)
    random.shuffle(val)

    new_train_json_path = os.path.abspath(f"checkpoints/train_{str(NUM_TRAJECTORY_TEMPLATES)}.json")
    new_val_json_path = os.path.abspath(f"checkpoints/val_{NUM_TRAJECTORY_TEMPLATES}.json")

    # Save train to a temp file
    with open(new_train_json_path, "w", encoding="utf-8") as f:
        json_data = json.dumps(train, ensure_ascii=False, indent=4)
        f.write(json_data)

    with open(new_val_json_path, "w", encoding="utf-8") as f:
        json_data = json.dumps(val, ensure_ascii=False, indent=4)
        f.write(json_data)


if __name__ == "__main__":
    main()
