DATA_PATH=/mnt/data/for_tllava/TinyLlava_text/blip_laion_cc_sbu_558k.json #pretrain annotation file path
IMAGE_PATH=/mnt/data/llava/dataset/llava/llava_pretrain #pretrain image dir
LLM_VERSION=/mnt/weight/HuggingFace/Qwen2-0.5B
VT_VERSION2="" #if you are not using mof vision tower, keep it empty
CN_VERSION=mlp2x_gelu #connector type, other options are: qformer, resampler, etc
VERSION=base_M3-04 #experiment name for recording different runnings
TRAIN_RECIPE=common #training recipes, other options are: lora, qlora
MODEL_MAX_LENGTH=2048 #max model length for llm
CONV_VERSION=qwen2_base
FINETUNE_DATA_PATH=/mnt/data/for_tllava/TinyLlava_text/llava_v1_5_mix665k_valid.json
FINETUNE_IMAGE_PATH=/mnt/data/llava/dataset


VT_VERSION=/mnt/weight/HuggingFace/google-siglip-so400m-patch14-384
VT_NAME=siglip-so400m-patch14-384

LLM_NAME=$(basename "$LLM_VERSION")



OUTPUT_DIR_PRE=/mnt/weight/checkpoints/llava_factory_Qwen/Qwen2/04-e3-tllava-${LLM_NAME}-${VT_NAME}-${VERSION}-pretrain_fa
OUTPUT_DIR_FT=/mnt/weight/checkpoints/llava_factory_Qwen/Qwen2/04-pe3_tllava-${LLM_NAME}-${VT_NAME}-${VERSION}-finetune_fa


bash scripts/train/train_qwen/pretrain_qwen.sh "$DATA_PATH" "$IMAGE_PATH" "$LLM_VERSION" "$VT_VERSION" "$VT_VERSION2" "$CN_VERSION" "$VERSION" "$TRAIN_RECIPE" "$MODEL_MAX_LENGTH" "$OUTPUT_DIR_PRE"
bash scripts/train/train_qwen/finetune_qwen.sh "$FINETUNE_DATA_PATH" "$FINETUNE_IMAGE_PATH" "$LLM_VERSION" "$VT_VERSION" "$VT_VERSION2" "$CN_VERSION" "$CONV_VERSION" "$VERSION" "$TRAIN_RECIPE" "$MODEL_MAX_LENGTH" "$OUTPUT_DIR_PRE" "$OUTPUT_DIR_FT"
