#!/bin/bash

# -----------------------MODEL SETTING------------------------
vision_model="facebook/dinov2-base"
# vision_model="facebook/dinov2-large"
# vision_model="ijepa-huge"
# vision_model="facebook/dinov2-giant"
# vision_model="openai/clip-vit-large-patch14"
# vision_model="dinov1-vitb16"
# vision_model="dinov1-resnet"
# vision_model="facebook/dinov2-giant"
# vision_model="mae-large"
# vision_model="aim-1B"
# vision_model="aim-600M"
# vision_model="ibot-large"

# text_model="sentence-transformers/all-mpnet-base-v2"
text_model="Alibaba-NLP/gte-base-en-v1.5" 
# text_model="openai/clip-vit-large-patch14"
# text_model="Alibaba-NLP/gte-base-en-v1.5" 
# text_model="Alibaba-NLP/gte-Qwen2-1.5B-instruct"
# text_model="Alibaba-NLP/gte-Qwen2-7B-instruct"
# text_model="nvidia/NV-Embed-v2"
# ------------------------------------------------------------ 
  
CKPT="/home/doldol9080/VLM/kt/SAIL/logs/SAILModel_baseline_siglip_ep10_25_gete-base-en-v1.5_dinov2-base/checkpoints/epoch_10.pt" # path to the checkpoint .pt file, make sure the vision and text model match the checkpoint
DATASET_ROOT_DIR="/home/dataset"


# imagenetv1 COCO winoground MMVP
for task in COCO
do
    # check if the checkpoint exists
    if [ ! -f $CKPT ]; then
        echo "Checkpoint not found: $CKPT"
        continue
    fi
    echo "########################################################"
    echo "Evaluating checkpoint: $CKPT"

    python eval.py \
        --head-weights-path $CKPT \
        --task $task \
        --dataset_root_dir $DATASET_ROOT_DIR \
        --batch_size 32 \
        --seg_task_config evaluation/ClearCLIP/configs/cfg_coco_stuff164k_SAIL.py \
        --agg_mode concat \
        --width_factor 8 \
        --overwrite \
        --vision-model $vision_model \
        --text-model $text_model \
        --sail_model \
        # --sharelock \

done
