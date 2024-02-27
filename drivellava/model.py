import os
import re
import sys
from io import BytesIO
from typing import List

import requests
import torch
from PIL import Image


def image_parser(args):
    out = args.image_file.split(args.sep)
    return out


def load_image(image_file):
    if image_file.startswith("http") or image_file.startswith("https"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image


def load_images(image_files):
    out = []
    for image_file in image_files:
        image = load_image(image_file)
        out.append(image)
    return out


class DriveLLaVA:
    def __init__(self, args):

        LLAVA_PATH = os.path.abspath("./LLaVA")

        if LLAVA_PATH not in sys.path:
            sys.path.append(LLAVA_PATH)

        from llava.mm_utils import get_model_name_from_path
        from llava.model.builder import load_pretrained_model
        from llava.utils import disable_torch_init

        # Model Initialization
        # Assuming this function disables initialization in PyTorch
        disable_torch_init()

        self.model_name = get_model_name_from_path(args.model_path)
        self.tokenizer, self.model, self.image_processor, self.context_len = (
            load_pretrained_model(
                args.model_path,
                args.model_base,
                self.model_name,
                load_8bit=True,
            )
        )

        # Infer conversation mode based on model name
        if "llama-2" in self.model_name.lower():
            self.conv_mode = "llava_llama_2"
        elif "mistral" in self.model_name.lower():
            self.conv_mode = "mistral_instruct"
        elif "v1.6-34b" in self.model_name.lower():
            self.conv_mode = "chatml_direct"
        elif "v1" in self.model_name.lower():
            self.conv_mode = "llava_v1"
        elif "mpt" in self.model_name.lower():
            self.conv_mode = "mpt"
        else:
            self.conv_mode = "llava_v0"

        if args.conv_mode is not None and self.conv_mode != args.conv_mode:
            print(
                f"[WARNING] the auto inferred conversation mode is "
                f"{self.conv_mode}, while `--conv-mode` is {args.conv_mode}, "
                f"using {args.conv_mode}"
            )
            self.conv_mode = args.conv_mode

        self.args = args

    def run(self, query: str, image_files: List[str]):

        from llava.constants import (
            DEFAULT_IM_END_TOKEN,
            DEFAULT_IM_START_TOKEN,
            DEFAULT_IMAGE_TOKEN,
            IMAGE_PLACEHOLDER,
            IMAGE_TOKEN_INDEX,
        )
        from llava.conversation import conv_templates
        from llava.mm_utils import process_images, tokenizer_image_token

        # Process query
        qs = query
        image_token_se = (
            DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
        )
        print('self.model.config.mm_use_im_start_end', self.model.config.mm_use_im_start_end)
        if IMAGE_PLACEHOLDER in qs:
            if self.model.config.mm_use_im_start_end:
                qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
            else:
                qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
        else:
            if self.model.config.mm_use_im_start_end:
                qs = image_token_se + "\n" + qs
            else:
                qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

        # Prepare conversation
        conv = conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        # Process images
        # image_files = image_parser(self.args)
        images = load_images(image_files)
        image_sizes = [x.size for x in images]
        images_tensor = process_images(
            images, self.image_processor, self.model.config
        ).to(self.model.device, dtype=torch.float16)

        # Tokenize prompt
        input_ids = (
            tokenizer_image_token(
                prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
            )
            .unsqueeze(0)
            .to(self.model.device)
        )

        # Inference
        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=images_tensor,
                image_sizes=image_sizes,
                do_sample=True if self.args.temperature > 0 else False,
                temperature=self.args.temperature,
                top_p=self.args.top_p,
                num_beams=self.args.num_beams,
                max_new_tokens=self.args.max_new_tokens,
                use_cache=True,
            )

        outputs = self.tokenizer.batch_decode(
            output_ids, skip_special_tokens=True
        )[0].strip()
        print(outputs)

        return outputs
