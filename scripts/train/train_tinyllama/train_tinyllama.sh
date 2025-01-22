DATA_PATH=/mnt/data/for_tllava/TinyLlava_text/blip_laion_cc_sbu_558k.json #pretrain annotation file path
IMAGE_PATH=/mnt/data/llava/dataset/llava/llava_pretrain #pretrain image dir
LLM_VERSION=/mnt/weight/HuggingFace/TinyLlama-1.1B-Chat-v1.0
VT_VERSION2="" #if you are not using mof vision tower, keep it empty
CN_VERSION=mlp2x_gelu #connector type, other options are: qformer, resampler, etc
VERSION=base_M3-04 #experiment name for recording different runnings
TRAIN_RECIPE=common #training recipes, other options are: lora, qlora
MODEL_MAX_LENGTH=2048 #max model length for llm
CONV_VERSION=tinyllama
FINETUNE_DATA_PATH=/mnt/data/for_tllava/TinyLlava_text/llava_v1_5_mix665k_valid.json
FINETUNE_IMAGE_PATH=/mnt/data/llava/dataset

VT_VERSION=/mnt/weight/HuggingFace/google-siglip-so400m-patch14-384
VT_NAME=siglip-so400m-patch14-384

LLM_NAME=$(basename "$LLM_VERSION")



OUTPUT_DIR_PRE=/mnt/weight/checkpoints/llava_factory_tllama/tllama_base_M3-04/clip/04-e10_tllava-${LLM_NAME}-${VT_NAME}-${VERSION}-pretrain_fa
OUTPUT_DIR_FT=/mnt/weight/checkpoints/llava_factory_tllama/tllama_base_M3-04/clip/04-pe10_tllava-${LLM_NAME}-${VT_NAME}-${VERSION}-finetune_fa


bash scripts/train/train_tinyllama/pretrain_tllama.sh "$DATA_PATH" "$IMAGE_PATH" "$LLM_VERSION" "$VT_VERSION" "$VT_VERSION2" "$CN_VERSION" "$VERSION" "$TRAIN_RECIPE" "$MODEL_MAX_LENGTH" "$OUTPUT_DIR_PRE"
bash scripts/train/train_tinyllama/finetune_tllama.sh "$FINETUNE_DATA_PATH" "$FINETUNE_IMAGE_PATH" "$LLM_VERSION" "$VT_VERSION" "$VT_VERSION2" "$CN_VERSION" "$CONV_VERSION" "$VERSION" "$TRAIN_RECIPE" "$MODEL_MAX_LENGTH" "$OUTPUT_DIR_PRE" "$OUTPUT_DIR_FT"
