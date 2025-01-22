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

python -m tinyllava.eval.model_vqa_pope \
    --model-path $MODEL_PATH \
    --question-file $EVAL_DIR/pope/llava_pope_test.jsonl \
    --image-folder $EVAL_DIR/pope/val2014 \
    --answers-file $EVAL_DIR/pope/answers/$MODEL_NAME.jsonl \
    --temperature 0 \
    --conv-mode $CONV_MODE

python tinyllava/eval/eval_pope.py \
    --annotation-dir $EVAL_DIR/pope/coco \
    --question-file $EVAL_DIR/pope/llava_pope_test.jsonl \
    --result-file $EVAL_DIR/pope/answers/$MODEL_NAME.jsonl