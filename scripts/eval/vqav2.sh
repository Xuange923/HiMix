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


gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

#SPLIT="llava_vqav2_mscoco_test-dev2015"
#
#MODEL_PATH="/mnt/data/sata/yinghu/checkpoints/llava_factory/tiny-llava-phi_files-2-clip-vit-large-patch14-336-baseline-finetune/"
#MODEL_NAME="tiny-llava-phi_files-2-clip-vit-large-patch14-336-baseline-finetune2"
#EVAL_DIR="/home/ai/data/llava/dataset/eval"

SPLIT="llava_vqav2_mscoco_test-dev2015"




for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m tinyllava.eval.model_vqa_loader \
        --model-path $MODEL_PATH \
        --question-file $EVAL_DIR/vqav2/$SPLIT.jsonl \
        --image-folder $EVAL_DIR/vqav2/test2015 \
        --answers-file $EVAL_DIR/vqav2/answers/$SPLIT/$MODEL_NAME/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --conv-mode $CONV_MODE &
done

wait

output_file=$EVAL_DIR/vqav2/answers/$SPLIT/$MODEL_NAME/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

 Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat $EVAL_DIR/vqav2/answers/$SPLIT/$MODEL_NAME/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

python scripts/convert_vqav2_for_submission.py --split $SPLIT --ckpt $MODEL_NAME --dir $EVAL_DIR/vqav2
