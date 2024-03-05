"""
Trains LLAVA model on the cumulative dataset.
"""

import json
import os
import random
import subprocess
import sys
from typing import Dict, List

from drivellava.constants import COMMAVQ_DIR
from drivellava.trajectory_encoder import NUM_TRAJECTORY_TEMPLATES


def load_json_dataset(
    json_list: List[str],
):
    data = []
    for json_path in json_list:
        with open(json_path, "r", encoding="utf-8") as f:
            loaded = json.load(f)
            data.extend(loaded)

    return data


def load_json_dataset_balanced(
    json_list: List[str],
):

    data = []
    for json_path in json_list:
        with open(json_path, "r", encoding="utf-8") as f:
            loaded = json.load(f)
            data.extend(loaded)

    # Balance by the class given by data[index]["conversations"][1]["value"]
    class_dist: Dict[str, int] = {}
    for index in range(len(data)):
        class_name = data[index]["conversations"][1]["value"]
        if class_name in class_dist:
            class_dist[class_name] += 1
        else:
            class_dist[class_name] = 1

    min_class = min(class_dist.values())
    max_class = max(class_dist.values())
    mean_class = sum(class_dist.values()) / len(class_dist)
    std_class = sum(
        [(x - mean_class) ** 2 for x in class_dist.values()]
    ) / len(class_dist)
    std_class = std_class**0.5
    print(
        f"Min class: {min_class}, Max class: {max_class}, "
        f"Mean class: {mean_class}, Std class: {std_class}"
    )

    threshold = min_class

    final_data = []
    final_dist: Dict[str, int] = {}
    for index in range(len(data)):
        class_name = data[index]["conversations"][1]["value"]
        if class_name in final_dist:
            final_dist[class_name] += 1
        else:
            final_dist[class_name] = 1
        if final_dist[class_name] < threshold:
            final_data.append(data[index])

    return final_data


def main():

    train_json_path = os.path.join(
        COMMAVQ_DIR, f"train_{str(NUM_TRAJECTORY_TEMPLATES)}.json"
    )
    val_json_path = os.path.join(
        COMMAVQ_DIR, f"val_{str(NUM_TRAJECTORY_TEMPLATES)}.json"
    )

    train = load_json_dataset_balanced(
        [
            train_json_path,
        ],
    )
    val = load_json_dataset(
        [
            val_json_path,
        ],
    )

    print(f"Train: {len(train)}")
    print(f"Val: {len(val)}")

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

    # Assign paths to variables
    WORKING_DIR = os.path.abspath("./LLaVA/")
    DEEPSPEED_SCRIPT = "deepspeed llava/train/train_mem.py"
    DEEPSPEED_JSON = os.path.abspath("./config/zero3.json")
    MODEL_NAME = "liuhaotian/llava-v1.5-7b"
    DATA_PATH = new_train_json_path  # Replace with your JSON data path
    # VAL_DATA_PATH = new_val_json_path
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
        --version llava_llama_2 \
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
        --num_train_epochs 1 \
        --per_device_train_batch_size 16 \
        --per_device_eval_batch_size 4 \
        --gradient_accumulation_steps 1 \
        --evaluation_strategy "steps" \
        --eval_steps 50 \
        --save_strategy "steps" \
        --save_steps 50 \
        --save_total_limit 1 \
        --learning_rate 2e-8 \
        --weight_decay 0. \
        --warmup_ratio 0.03 \
        --lr_scheduler_type "cosine" \
        --logging_steps 1 \
        --tf32 True \
        --model_max_length 2048 \
        --gradient_checkpointing True \
        --dataloader_num_workers 4 \
        --lazy_preprocess True \
        --report_to wandb
    """

    print(finetune_script)

    # Run the command in WORKING_DIR
    subprocess.run(finetune_script, cwd=WORKING_DIR, shell=True)


if __name__ == "__main__":
    main()
