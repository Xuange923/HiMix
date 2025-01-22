#!/bin/bash

if [ $# -eq 4 ]; then
    MODEL_PATH="$1"
    MODEL_NAME="$2"
    EVAL_DIR="$3"
    CONV_MODE="$4"
else
  MODEL_PATH="/path/to/your/model"
  MODEL_NAME="your_model_name"
  EVAL_DIR="/path/to/your/eval_dir"
  CONV_MODE="qwen2_base"  # qwen2_base, tinyllama,llama-3
fi

python -m tinyllava.eval.model_vqa_loader \
    --model-path $MODEL_PATH \
    --question-file $EVAL_DIR/textvqa/llava_textvqa_val_v051_ocr.jsonl \
    --image-folder /mnt/mmdata/OCR/TextOCR/train \
    --answers-file $EVAL_DIR/textvqa/answers/$MODEL_NAME.jsonl \
    --temperature 0 \
    --conv-mode $CONV_MODE


python -m tinyllava.eval.eval_textvqa \
    --annotation-file $EVAL_DIR/textvqa/TextVQA_0.5.1_val.json \
    --result-file $EVAL_DIR/textvqa/answers/$MODEL_NAME.jsonl

