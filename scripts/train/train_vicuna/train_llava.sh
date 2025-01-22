
LLM_VERSION=/mnt/weight/HuggingFace/vicuna-7b-v1.5
VT_VERSION2="" #if you are not using mof vision tower, keep it empty
CN_VERSION=mlp2x_gelu #connector type, other options are: qformer, resampler, etc
VERSION=OneVision_M3-04 #experiment name for recording different runnings
TRAIN_RECIPE=common #training recipes, other options are: lora, qlora
MODEL_MAX_LENGTH=2048 #max model length for llm
CONV_VERSION=tinyllama


VT_VERSION=/mnt/weight/HuggingFace/google-siglip-so400m-patch14-384
VT_NAME=siglip-so400m-patch14-384


LLM_NAME=$(basename "$LLM_VERSION")


DATA_PATH=/mnt/data/llava_instruct/merged/mid_stage_merged.json #pretrain annotation file path
IMAGE_PATH=/mnt/data/LLaVA-OneVision/Image_Data #pretrain image dir
FINETUNE_DATA_PATH=/mnt/data/llava_instruct/merged/Single_Image_merged_sampled.json
FINETUNE_IMAGE_PATH=/mnt/data/LLaVA-OneVision/Image_Data


OUTPUT_DIR_PRE=/mnt/weight/checkpoints/llava_factory_llava/llava_base_M3-04/04-e1_tllava-${LLM_NAME}-${VT_NAME}-${VERSION}-pretrain_fa
OUTPUT_DIR_FT=/mnt/weight/checkpoints/llava_factory_llava/llava_base_M3-04/04-pe1_tllava-${LLM_NAME}-${VT_NAME}-${VERSION}-finetune_fa



bash scripts/train/train_vicuna/pretrain_vicuna.sh "$DATA_PATH" "$IMAGE_PATH" "$LLM_VERSION" "$VT_VERSION" "$VT_VERSION2" "$CN_VERSION" "$VERSION" "$TRAIN_RECIPE" "$MODEL_MAX_LENGTH" "$OUTPUT_DIR_PRE"
bash scripts/train/train_llava/finetune_llava.sh "$FINETUNE_DATA_PATH" "$FINETUNE_IMAGE_PATH" "$LLM_VERSION" "$VT_VERSION" "$VT_VERSION2" "$CN_VERSION" "$CONV_VERSION" "$VERSION" "$TRAIN_RECIPE" "$MODEL_MAX_LENGTH" "$OUTPUT_DIR_PRE" "$OUTPUT_DIR_FT"
