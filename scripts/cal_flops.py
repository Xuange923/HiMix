# Code from https://github.com/MrYxJ/calculate-flops.pytorch?tab=readme-ov-file

import argparse
import json
import os
import torch
from PIL import Image
from calflops import calculate_flops
from transformers import AutoModel, AutoTokenizer
from tinyllava.utils import *
from tinyllava.data import *
from tinyllava.model import *


def load_single_data(question_file, image_folder, text_processor, image_processor, seq_len=64):
    seq_len += 1
    with open(question_file, "r") as f:
        questions = [json.loads(line) for line in f]

    line = questions[0]
    image_file = line["image"]

    image = Image.open(os.path.join(image_folder, image_file)).convert('RGB')
    image_tensor = image_processor(image)

    qs = DEFAULT_IMAGE_TOKEN + '\n' + "AA"
    msg = Message()
    msg.add_message(qs)
    result = text_processor(msg.messages, mode='eval')
    input_ids = result['input_ids']

    if isinstance(input_ids, torch.Tensor):
        input_ids = input_ids.tolist()

        # If the length of input_ids is less than seq_len, pad it to seq_len.
    if len(input_ids) < seq_len:
        fill_start_index = max(len(input_ids) - 5, 0)
        padding_length = seq_len - len(input_ids)

        input_ids = (
                input_ids[:fill_start_index] +
                [51] * padding_length +
                input_ids[fill_start_index:]
        )

    input_ids = input_ids[:seq_len]
    input_ids = torch.tensor(input_ids)

    return input_ids, image_tensor, image.size

# def load_single_data(question_file, image_folder, text_processor, image_processor):
#     with open(question_file, "r") as f:
#         questions = [json.loads(line) for line in f]
#
#     line = questions[0]
#     image_file = line["image"]
#     qs = line["text"]
#
#     image = Image.open(os.path.join(image_folder, image_file)).convert('RGB')
#     image_tensor = image_processor(image)
#
#     qs = DEFAULT_IMAGE_TOKEN + '\n' + qs
#     msg = Message()
#     msg.add_message(qs)
#     result = text_processor(msg.messages, mode='eval')
#     input_ids = result['input_ids']
#
#     return input_ids, image_tensor, image.size


def eval_model(args):
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model, tokenizer, image_processor, context_len = load_pretrained_model(model_path)

    text_processor = TextPreprocess(tokenizer, args.conv_mode)
    data_args = model.config
    image_processor = ImagePreprocess(image_processor, data_args)

    input_ids, image_tensor, image_sizes = load_single_data(args.question_file, args.image_folder, text_processor, image_processor)

    inputs = {
        'input_ids': input_ids.unsqueeze(0),
        'images': image_tensor.unsqueeze(0),
    }
    model.to(device='cuda')

    flops, macs, params = calculate_flops(model=model, kwargs=inputs, print_results=True)
    print(f"Model FLOPs: {flops}   MACs: {macs}   Params: {params}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--image-folder", type=str,
                        default="/mnt/data/for_tllava/eval/vqav2/test2015")
    parser.add_argument("--question-file", type=str,
                        default="/mnt/data/for_tllava/eval/vqav2/llava_vqav2_mscoco_test-dev2015.jsonl")
    parser.add_argument("--max_new_tokens", type=int, default=1280)
    parser.add_argument("--model-path", type=str,
                        default="/path/to/your/model")
    parser.add_argument("--conv_mode", type=str, default="tinyllama") # tinyllama,llama-3,qwen2_base



    args = parser.parse_args()
    eval_model(args)
