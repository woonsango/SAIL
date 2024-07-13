#!/bin/bash
#SBATCH --job-name=eval
#SBATCH --partition=short-unkillable                           # Ask for unkillable job
#SBATCH --cpus-per-task=24                             # Ask for 2 CPUs
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100l:1
#SBATCH --ntasks-per-node=1                                  # Ask for 1 GPU
#SBATCH --mem=128G           
#SBATCH --time=3:00:00                                    
#SBATCH --output=./slurm_logs/eval/output-%j.txt
#SBATCH --error=./slurm_logs/eval/error-%j.txt 

# vision_model="facebook/dinov2-base"
vision_model="facebook/dinov2-large"

# text_model="sentence-transformers/all-mpnet-base-v2"
text_model="Alibaba-NLP/gte-large-en-v1.5"
# text_model="Alibaba-NLP/gte-Qwen2-1.5B-instruct"

export HF_AUTH_TOKEN="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"

for train in llava_vflan_gtedinoL_bs_32768_lion_mean
do
    for task in imagenet winoground coco
    do
        for epoch in {10..300..10}; 
        do
            checkpoint_path="./logs/${train}/checkpoints/epoch_${epoch}.pt"
            # check if the checkpoint exists
            if [ ! -f $checkpoint_path ]; then
                echo "Checkpoint not found: $checkpoint_path"
                continue
            fi
            echo "Evaluating checkpoint: $checkpoint_path"

            python eval.py \
                --head-weights-path $checkpoint_path \
                --task $task \
                --vision-model $vision_model \
                --text-model $text_model \
                --linear-align 
        done
    done
done