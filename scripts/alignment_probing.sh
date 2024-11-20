#!/bin/bash
#SBATCH --job-name=NV2CLIP
#SBATCH --partition=unkillable         # Ask for unkillable job
#SBATCH --cpus-per-task=4                             # Ask for 2 CPUs
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100l:1
#SBATCH --ntasks-per-node=1                                  # Ask for 1 GPU
#SBATCH --mem=32G           
#SBATCH --time=9:00:00                                    
#SBATCH --output=./slurm_logs/train/abaltion/%x_%j_%A_%a_${data}.out
#SBATCH --error=./slurm_logs/train/abaltion/%x_%j_%A_%a_${data}.err 

module load miniconda/3
conda init
conda activate openflamingo


# ----------------------TRAIN SETTING------------------------


text_embedding_list="data/tensor_data/text_embedding/NV-Embed-v2/dreamclipcc3m_raw_caption"
image_embedding_list="data/tensor_data/image_embedding/dinov2-large/dreamclipcc3m"
extra_text_embedding_list="data/tensor_data/text_embedding/NV-Embed-v2/dreamclipcc3m_longSV_captions"
output_name="alignment_probing_dinov2l_nv2"



epoch_num=100
logit_scale=20
logit_bias=-10
alpha=0.995
lr=1e-5
bs=32768
d=2048
width_factor=8
linear_type="linear"



python main.py \
    --text-embedding-list $text_embedding_list \
    --image-embedding-list $image_embedding_list \
    --extra-text-embedding-list $extra_text_embedding_list \
    --val-frequency 2 \
    --dataset-type embedding \
    --seed 42 \
    --resume latest \
    --save-frequency 2 \
    --report-to wandb \
    --batch-size $bs \
    --lr $lr \
    --epochs $epoch_num \
    --workers 0 \
    --wd 1e-07 \
    --beta1 0.9 \
    --beta2 0.99 \
    --target-dimension $d \
    --linear-type $linear_type \
    --log-every-n-steps 5 \
    --wandb-project-name alignment_probing \
    --name $output_name \
    --logit_scale $logit_scale \
    --logit_bias $logit_bias \
    --siglip \
    --width-factor $width_factor



if [ $? -ne 0 ]; then
    echo "Training failed. Checking for checkpoints..."
    
    if ls ./logs/$output_name/checkpoints/*.pt 1> /dev/null 2>&1; then
        echo "Checkpoint file(s) found. Keeping the log directory."
    else
        echo "No checkpoint files found. Cleaning up ./logs/${output_name}"
        rm -rf ./logs/$output_name
    fi
fi

