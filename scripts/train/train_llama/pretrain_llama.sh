#!/bin/bash

if [ $# -ne 10 ]; then
    echo "Usage: $0 <DATA_PATH> <IMAGE_PATH> <LLM_VERSION> <VT_VERSION> <VT_VERSION2> <CN_VERSION> <VERSION> <TRAIN_RECIPE> <MODEL_MAX_LENGTH>"
    exit 1
fi


# Assign the arguments to variables
DATA_PATH="$1"
IMAGE_PATH="$2"
LLM_VERSION="$3"
VT_VERSION="$4"
VT_VERSION2="$5"
CN_VERSION="$6"
VERSION="$7"
TRAIN_RECIPE="$8"
MODEL_MAX_LENGTH="$9"
OUTPUT_DIR="${10}"

RUN_NAME=$(basename $OUTPUT_DIR)

mkdir -p $OUTPUT_DIR



deepspeed --include localhost:0,1,2,3 --master_port 29501 tinyllava/train/train.py \
    --deepspeed ./scripts/zero3.json \
    --data_path  $DATA_PATH\
    --image_folder $IMAGE_PATH \
    --is_multimodal True \
    --conv_version pretrain \
    --model_name_or_path $LLM_VERSION \
    --vision_tower $VT_VERSION \
    --vision_tower2 "$VT_VERSION2" \
    --connector_type $CN_VERSION \
    --mm_vision_select_layer -2 \
    --image_aspect_ratio square \
    --attn_implementation flash_attention_2 \
    --fp16 False \
    --bf16 True \
    --training_recipe $TRAIN_RECIPE \
    --tune_type_llm part \
    --tune_type_vision_tower frozen \
    --tune_vision_tower_from_layer 0 \
    --tune_type_connector full \
    --output_dir $OUTPUT_DIR\
    --num_train_epochs 3 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "no" \
    --save_strategy "epoch" \
    --save_steps 1000 \
    --save_total_limit 1 \
    --learning_rate 5e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 False \
    --model_max_length $MODEL_MAX_LENGTH \
    --gradient_checkpointing True \
    --dataloader_num_workers 8 \
    --lazy_preprocess True \
    --report_to tensorboard \
    --tokenizer_use_fast False \
    --run_name $RUN_NAME

