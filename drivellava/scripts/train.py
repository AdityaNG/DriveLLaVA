"""
Trains LLAVA model on the cumulative dataset.
"""

import json
import os
import subprocess
import sys
from typing import List

from drivellava.constants import ENCODED_JSON, VAL_ENCODED_JSON


def load_json_dataset(
    json_list: List[str],
):
    data = []
    for json_path in json_list:
        with open(json_path, "r") as f:
            data.extend(json.load(f))

    return data


def main():
    train = load_json_dataset(ENCODED_JSON)
    val = load_json_dataset(VAL_ENCODED_JSON)

    train_json_path = os.path.abspath("checkpoints/train.json")

    # Save train to a temp file
    with open(train_json_path, "w") as f:
        json.dump(train, f)

    print(f"Train: {len(train)}")
    print(f"Val: {len(val)}")

    # Assign paths to variables
    WORKING_DIR = os.path.abspath("./LLaVA/")
    DEEPSPEED_SCRIPT = "deepspeed llava/train/train_mem.py"
    DEEPSPEED_JSON = "./scripts/zero3.json"
    MODEL_NAME = "liuhaotian/llava-v1.5-7b"
    DATA_PATH = train_json_path  # Replace with your JSON data path
    IMAGE_FOLDER = "/"  # Replace with your image folder path
    VISION_TOWER = "openai/clip-vit-large-patch14-336"
    OUTPUT_DIR = os.path.abspath("./checkpoints")

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
        --save_steps 50000 \
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
