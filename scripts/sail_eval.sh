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
conda activate openflamingo
module load cudatoolkit/12.4.0



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
# text_model="Alibaba-NLP/gte-large-en-v1.5" 
# text_model="openai/clip-vit-large-patch14"
# text_model="Alibaba-NLP/gte-base-en-v1.5" 
# text_model="Alibaba-NLP/gte-Qwen2-1.5B-instruct"
# text_model="Alibaba-NLP/gte-Qwen2-7B-instruct"
text_model="nvidia/NV-Embed-v2"
# ------------------------------------------------------------ 
  
checkpoint_path="" # path to the checkpoint .pt file, make sure the vision and text model match the checkpoint

# imagenetv1 COCO winoground MMVP
for task in imagenetv1 COCO winoground MMVP
do
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
        --text-model $text_model \
        --batch_size 32 \
        --seg_task_config evaluation/ClearCLIP/configs/cfg_coco_stuff164k_SAIL.py \
        --agg_mode concat \
        --width_factor 8 \
        --sharelock
done
