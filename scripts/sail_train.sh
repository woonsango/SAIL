#!/bin/bash
# ----------------------TRAIN SETTING------------------------

epoch_num=10
lr=1e-5
bs=32768
d=1024
width_factor=8
logit_scale=20
logit_bias=-10

text_model="Alibaba-NLP/gte-base-en-v1.5"
vision_model="facebook/dinov2-base"

text_embedding_list="data/tensor_data/text_embedding/gte-base-en-v1.5/coco2017_captions" 
image_embedding_list="data/tensor_data/image_embedding/dinov2-base/coco2017_concat"
# extra_text_embedding_list="data/tensor_data/text_embedding/NV-Embed-v2/yfcc15m_shortSV_captions data/tensor_data/text_embedding/NV-Embed-v2/dreamclipcc3m_longSV_captions data/tensor_data/text_embedding/NV-Embed-v2/dreamclipcc12mhf_shortSV_captions"
output_name="SAILModel_baseline_siglip_ep10_25_gete-base-en-v1.5_dinov2-base"
# ------------------------------------------------------------

DATASET_ROOT_DIR="/home/dataset"

python main.py \
    --text-embedding-list $text_embedding_list \
    --image-embedding-list $image_embedding_list \
    --val-frequency 1 \
    --dataset-type embedding \
    --seed 42 \
    --resume latest \
    --save-frequency 2 \
    --report-to wandb \
    --batch-size $bs \
    --lr $lr \
    --epochs $epoch_num \
    --workers 0 \
    --optimizer lion \
    --wd 1e-4 \
    --target-dimension $d \
    --linear-type star \
    --log-every-n-steps 5 \
    --wandb-project-name sail_train \
    --name $output_name \
    --logit_scale $logit_scale \
    --logit_bias $logit_bias \
    --text-model $text_model \
    --vision-model $vision_model \
    --siglip \
    --sail_model \
    # --sharelock \
    # --extra-text-embedding-list $extra_text_embedding_list \



if [ $? -ne 0 ]; then
    echo "Training failed. Checking for checkpoints..."
    
    if ls ./logs/$output_name/checkpoints/*.pt 1> /dev/null 2>&1; then
        echo "Checkpoint file(s) found. Keeping the log directory."
    else
        echo "No checkpoint files found. Cleaning up ./logs/${output_name}"
        rm -rf ./logs/$output_name
    fi
fi
