#!/bin/bash
#SBATCH --job-name=eval
#SBATCH --partition=long                           # Ask for unkillable job
#SBATCH --cpus-per-task=4                             # Ask for 2 CPUs
#SBATCH --nodes=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --ntasks-per-node=1                                  # Ask for 1 GPU
#SBATCH --mem=64G           
#SBATCH --time=0:30:00                                    
#SBATCH --output=./slurm_logs/eval/output-%j.txt
#SBATCH --error=./slurm_logs/eval/error-%j.txt 


module load miniconda/3
conda init
# conda activate openflamingo

export HF_AUTH_TOKEN="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"



# -----------------------MODEL SETTING------------------------
# vision_model="facebook/dinov2-base"
vision_model="facebook/dinov2-large"
# vision_model="facebook/dinov2-giant"
# vision_model="dinov1-vitb16"
# vision_model="dinov1-resnet"
# vision_model="facebook/dinov2-giant"
# vision_model="mae-large"
# vision_model="aim-1B"
# vision_model="aim-600M"
# vision_model="ibot-large"

# text_model="sentence-transformers/all-mpnet-base-v2"
text_model="Alibaba-NLP/gte-large-en-v1.5" 
# text_model="Alibaba-NLP/gte-base-en-v1.5" 
# text_model="Alibaba-NLP/gte-Qwen2-1.5B-instruct"
# text_model="Alibaba-NLP/gte-Qwen2-7B-instruct"
# text_model="nvidia/NV-Embed-v2"
# ------------------------------------------------------------ 
  
# for train in dreamclip30m_NV2dinoL_bs_32768_lion_mean_lr_1e-5_star7XL_d1024_scale20_bias-10_multi_postext_s2 14
for train in dreamclip30m_gtendinoL_bs_32768_lion_mean_lr_1e-5_star7XL_d1024_scale20_bias-10_multi_postext_s2
do 
    # for epoch in $(seq 8 2 100)
    for epoch in 30
    do
        # imagenetv1 imagenetv2 COCO winoground sugar_crepe MMVP
        for task in segmentation
        do
            checkpoint_path="./logs/${train}/checkpoints/epoch_${epoch}.pt"
            # check if the checkpoint exists
            if [ ! -f $checkpoint_path ]; then
                echo "Checkpoint not found: $checkpoint_path"
                continue
            fi
            echo "########################################################"
            echo "Evaluating checkpoint: $checkpoint_path"

            python eval.py \
                --head-weights-path $checkpoint_path \
                --task $task \
                --vision-model $vision_model \
<<<<<<< HEAD
                --text-model $text_model 
=======
                --text-model $text_model \
                --batch_size 64 \
                --seg_task_config /home/mila/l/le.zhang/scratch/light_align/evaluation/ClearCLIP/configs/cfg_coco_stuff164k_SAIL.py \
                --overwrite
        done
    done
done