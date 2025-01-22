#!/bin/bash

# vqav2_evaluation_server: https://eval.ai/web/challenges/challenge-page/830/my-submission
# mmvet_evaluation_server: https://huggingface.co/spaces/whyu/MM-Vet_Evaluator

MODEL_PATH="/path/to/your/model"
MODEL_NAME="your_model_name"
EVAL_DIR="/path/to/your/eval_dir"
CONV_MODE="qwen2_base"  # qwen2_base, tinyllama,llama-3



echo $MODEL_PATH
echo $MODEL_NAME


echo "***start*** mmmu"
CUDA_VISIBLE_DEVICES=0 bash scripts/eval/mmmu.sh "$MODEL_PATH" "$MODEL_NAME" "$EVAL_DIR" "$CONV_MODE"
if [ $? -ne 0 ]; then
    echo "mmmu.sh failed"
fi
echo "***end*** mmmu"

echo "***start*** mme"
CUDA_VISIBLE_DEVICES=0 bash scripts/eval/mme.sh "$MODEL_PATH" "$MODEL_NAME" "$EVAL_DIR" "$CONV_MODE"
if [ $? -ne 0 ]; then
   echo "mme.sh failed"
fi
echo "***end*** mme"



echo "***start*** pope"
CUDA_VISIBLE_DEVICES=0 bash scripts/eval/pope.sh "$MODEL_PATH" "$MODEL_NAME" "$EVAL_DIR" "$CONV_MODE"
if [ $? -ne 0 ]; then
    echo "pope.sh failed"
fi
echo "***end*** pope"


echo "***start*** textvqa"
CUDA_VISIBLE_DEVICES=0 bash scripts/eval/textvqa.sh "$MODEL_PATH" "$MODEL_NAME" "$EVAL_DIR" "$CONV_MODE"
if [ $? -ne 0 ]; then
    echo "textvqa.sh failed"
fi
echo "***end*** textvqa"

echo "***start*** mmvet"
CUDA_VISIBLE_DEVICES=0 bash scripts/eval/mmvet.sh "$MODEL_PATH" "$MODEL_NAME" "$EVAL_DIR" "$CONV_MODE"
if [ $? -ne 0 ]; then
    echo "mmvet.sh failed"
fi
echo "***end*** mmvet"


echo "***start*** gqa"
CUDA_VISIBLE_DEVICES=0,1 bash scripts/eval/gqa.sh "$MODEL_PATH" "$MODEL_NAME" "$EVAL_DIR" "$CONV_MODE"
if [ $? -ne 0 ]; then
    echo "gqa.sh failed"
fi
echo "***end*** gqa"

echo "***start*** vqav2"
CUDA_VISIBLE_DEVICES=0,1 bash scripts/eval/vqav2.sh "$MODEL_PATH" "$MODEL_NAME" "$EVAL_DIR" "$CONV_MODE"
if [ $? -ne 0 ]; then
    echo "vqav2.sh failed"
fi
echo "***end*** vqav2"















