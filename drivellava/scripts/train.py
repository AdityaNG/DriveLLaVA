"""
Trains LLAVA model on the cumulative dataset.
"""

import json
import os
import random
import subprocess
import sys
from typing import List

from drivellava.constants import COMMAVQ_DIR
from drivellava.trajectory_encoder import TrajectoryEncoder


def load_json_dataset(
    json_list: List[str],
    trajectory_encoder: TrajectoryEncoder,
):
    from drivellava.sparse_llava_dataset import get_drivellava_prompt

    data = []
    for json_path in json_list:
        with open(json_path, "r", encoding="utf-8") as f:
            loaded = json.load(f)
            for index in range(len(loaded)):
                assert len(loaded[index]["conversations"][1]["value"]) == 1
                loaded[index]["conversations"][0]["value"] = (
                    get_drivellava_prompt(trajectory_encoder)
                )
            data.extend(loaded)

    return data


def main():
    # train = load_json_dataset(ENCODED_JSON)
    # val = load_json_dataset(VAL_ENCODED_JSON)

    # train_json_path = os.path.abspath("checkpoints/train.json")
    # val_json_path = os.path.abspath("checkpoints/val.json")

    # # Save train to a temp file
    # with open(train_json_path, "w", encoding="utf-8") as f:
    #     json_data = json.dumps(train, ensure_ascii=False, indent=4)
    #     f.write(json_data)

    # with open(val_json_path, "w", encoding="utf-8") as f:
    #     json_data = json.dumps(val, ensure_ascii=False, indent=4)
    #     f.write(json_data)

    train_json_path = os.path.join(COMMAVQ_DIR, "train.json")
    val_json_path = os.path.join(COMMAVQ_DIR, "val.json")

    trajectory_encoder = TrajectoryEncoder()

    train = load_json_dataset(
        [
            train_json_path,
        ],
        trajectory_encoder,
    )
    val = load_json_dataset(
        [
            val_json_path,
        ],
        trajectory_encoder,
    )

    # Shuffle train and val
    random.shuffle(train)
    random.shuffle(val)

    new_train_json_path = os.path.abspath("checkpoints/train.json")
    new_val_json_path = os.path.abspath("checkpoints/val.json")

    # Save train to a temp file
    with open(new_train_json_path, "w", encoding="utf-8") as f:
        json_data = json.dumps(train, ensure_ascii=False, indent=4)
        f.write(json_data)

    with open(new_val_json_path, "w", encoding="utf-8") as f:
        json_data = json.dumps(val, ensure_ascii=False, indent=4)
        f.write(json_data)

    print(f"Train: {len(train)}")
    print(f"Val: {len(val)}")

    # Assign paths to variables
    WORKING_DIR = os.path.abspath("./LLaVA/")
    DEEPSPEED_SCRIPT = "deepspeed llava/train/train_mem.py"
    DEEPSPEED_JSON = os.path.abspath("./config/zero3.json")
    MODEL_NAME = "liuhaotian/llava-v1.5-7b"
    DATA_PATH = new_train_json_path  # Replace with your JSON data path
    IMAGE_FOLDER = os.path.expanduser(
        "~/Datasets/commavq"
    )  # Replace with your image folder path
    VISION_TOWER = "openai/clip-vit-large-patch14-336"
    OUTPUT_DIR = os.path.expanduser("~/Datasets/checkpoints")

    sys.path.append(WORKING_DIR)

    # Command to run the script
    finetune_script = f"""
    {DEEPSPEED_SCRIPT} \
        --lora_enable True --lora_r 128 --lora_alpha 256 \
        --mm_projector_lr 2e-5 \
        --deepspeed {DEEPSPEED_JSON} \
        --model_name_or_path {MODEL_NAME} \
        --version v1 \
        --data_path {DATA_PATH} \
        --image_folder {IMAGE_FOLDER} \
        --vision_tower {VISION_TOWER} \
        --mm_projector_type mlp2x_gelu \
        --mm_vision_select_layer -2 \
        --mm_use_im_start_end False \
        --mm_use_im_patch_token False \
        --image_aspect_ratio pad \
        --group_by_modality_length True \
        --bf16 True \
        --output_dir {OUTPUT_DIR} \
        --num_train_epochs 5 \
        --per_device_train_batch_size 16 \
        --per_device_eval_batch_size 4 \
        --gradient_accumulation_steps 1 \
        --evaluation_strategy "no" \
        --save_strategy "steps" \
        --save_steps 1000 \
        --save_total_limit 1 \
        --learning_rate 2e-4 \
        --weight_decay 0. \
        --warmup_ratio 0.03 \
        --lr_scheduler_type "cosine" \
        --logging_steps 1 \
        --tf32 True \
        --model_max_length 2048 \
        --gradient_checkpointing True \
        --dataloader_num_workers 4 \
        --lazy_preprocess True \
        --report_to wandb \
        --freeze_backbone \
        --freeze_mm_mlp_adapter
    """

    print(finetune_script)

    # Run the command in WORKING_DIR
    subprocess.run(finetune_script, cwd=WORKING_DIR, shell=True)


if __name__ == "__main__":
    main()
