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

python -m tinyllava.eval.model_vqa \
    --model-path $MODEL_PATH \
    --question-file $EVAL_DIR/mm-vet/llava-mm-vet.jsonl \
    --image-folder $EVAL_DIR/mm-vet/images \
    --answers-file $EVAL_DIR/mm-vet/answers/$MODEL_NAME.jsonl \
    --temperature 0 \
    --conv-mode $CONV_MODE

mkdir -p $EVAL_DIR/mm-vet/results

python scripts/convert_mmvet_for_eval.py \
    --src $EVAL_DIR/mm-vet/answers/$MODEL_NAME.jsonl \
    --dst $EVAL_DIR/mm-vet/results/$MODEL_NAME.json
