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

python -m tinyllava.eval.model_vqa_mmmu \
    --model-path $MODEL_PATH \
    --question-file $EVAL_DIR/MMMU/anns_for_eval.json \
    --image-folder $EVAL_DIR/MMMU/all_images \
    --answers-file $EVAL_DIR/MMMU/answers/$MODEL_NAME.jsonl \
    --temperature 0 \
    --conv-mode $CONV_MODE

python scripts/convert_answer_to_mmmu.py \
    --answers-file $EVAL_DIR/MMMU/answers/$MODEL_NAME.jsonl \
    --answers-output $EVAL_DIR/MMMU/answers/"$MODEL_NAME"_output.json

cd $EVAL_DIR/MMMU/eval

python main_eval_only.py --output_path $EVAL_DIR/MMMU/answers/"$MODEL_NAME"_output.json
