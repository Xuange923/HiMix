DATA_PATH=/mnt/data/for_tllava/TinyLlava_text/blip_laion_cc_sbu_558k.json #pretrain annotation file path
IMAGE_PATH=/mnt/data/llava/dataset/llava/llava_pretrain #pretrain image dir


LLM_VERSION=/mnt/weight/HuggingFace/Llama-3.2-1B
VT_VERSION2="" #if you are not using mof vision tower, keep it empty
CN_VERSION=mlp2x_gelu #connector type, other options are: qformer, resampler, etc
VERSION=base_M3-04 #experiment name for recording different runnings
TRAIN_RECIPE=common #training recipes, other options are: lora, qlora
MODEL_MAX_LENGTH=2048 #max model length for llm
CONV_VERSION=llama-3
FINETUNE_DATA_PATH=/mnt/data/for_tllava/TinyLlava_text/llava_v1_5_mix665k_valid.json
FINETUNE_IMAGE_PATH=/mnt/data/llava/dataset


VT_VERSION=/mnt/weight/HuggingFace/google-siglip-so400m-patch14-384
VT_NAME=siglip-so400m-patch14-384
#VT_VERSION=/mnt/weight/HuggingFace/dinov2-base
#VT_NAME=dinov2-base
#VT_VERSION=/mnt/weight/HuggingFace/TinyCLIP-ViT-40M-32-Text-19M-LAION400M
#VT_NAME=TinyCLIP-ViT-40M-32-Text-19M-LAION400M
#VT_VERSION=/mnt/weight/HuggingFace/TinyCLIP-ViT-39M-16-Text-19M-YFCC15M
#VT_NAME=TinyCLIP-ViT-39M-16
#VT_VERSION=/mnt/weight/HuggingFace/TinyCLIP-ViT-61M-32-Text-29M-LAION400M
#VT_NAME=TinyCLIP-ViT-61M-32
#VT_VERSION=/mnt/weight/HuggingFace/beit-base-patch16-224-pt22k-ft22k
#VT_NAME=beit-base-patch16-224-pt22k-ft22k
#VT_VERSION=/mnt/weight/HuggingFace/vit-base-patch16-224-in21k
#VT_NAME=vit-base-patch16-224-in21k
#VT_VERSION=/mnt/weight/HuggingFace/hiera-base-224-hf
#VT_NAME=hiera-base-224-hf

LLM_NAME=$(basename "$LLM_VERSION")



OUTPUT_DIR_PRE=/mnt/weight/checkpoints/llava_factory_llama/siglip/04-e1_tllava-${LLM_NAME}-${VT_NAME}-${VERSION}-pretrain_fa-test
OUTPUT_DIR_FT=/mnt/weight/checkpoints/llava_factory_llama/siglip/04-pe1_tllava-${LLM_NAME}-${VT_NAME}-${VERSION}-finetune_fa-test


bash scripts/train/train_llama/pretrain_llama.sh "$DATA_PATH" "$IMAGE_PATH" "$LLM_VERSION" "$VT_VERSION" "$VT_VERSION2" "$CN_VERSION" "$VERSION" "$TRAIN_RECIPE" "$MODEL_MAX_LENGTH" "$OUTPUT_DIR_PRE"
bash scripts/train/train_llama/finetune_llama.sh "$FINETUNE_DATA_PATH" "$FINETUNE_IMAGE_PATH" "$LLM_VERSION" "$VT_VERSION" "$VT_VERSION2" "$CN_VERSION" "$CONV_VERSION" "$VERSION" "$TRAIN_RECIPE" "$MODEL_MAX_LENGTH" "$OUTPUT_DIR_PRE" "$OUTPUT_DIR_FT"
