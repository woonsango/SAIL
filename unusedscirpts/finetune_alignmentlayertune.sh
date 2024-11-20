#!/bin/bash
#SBATCH --job-name=nv_ft_tunevlhead
#SBATCH --partition=short-unkillable    # Ask for unkillable job
#SBATCH --cpus-per-task=24                             # Ask for 2 CPUs
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:4
#SBATCH --ntasks-per-node=1                                  # Ask for 1 GPU
#SBATCH --mem=128G           
#SBATCH --time=3:00:00                                    
#SBATCH --output=./slurm_logs/train/llava1.5/%x_%j.out
#SBATCH --error=./slurm_logs/train/llava1.5/%x_%j.err 


module load miniconda/3
conda init
conda activate llava
module load cudatoolkit/12.1.1

deepspeed VLM_Training/train_mem.py \
    --deepspeed VLM_Training/zero3.json \
    --model_name_or_path lmsys/vicuna-7b-v1.5 \
    --version v1 \
    --data_path /network/scratch/q/qian.yang/llava-v1.5-7b/instruct_tuning_data/llava_v1_5_mix665k.json \
    --image_folder /network/scratch/q/qian.yang/llava-v1.5-7b/instruct_tuning_data/data \
    --target_dimension 1024 \
    --linear_type star \
    --vision_tower facebook/dinov2-large \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --num_train_epochs 1 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 400 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb \
    --vlhead_weights_path llava_checkpoints/llava_stage1_dreamclip30m_NV2dinoL_sequence/vlhead.bin \
    --pretrain_mm_mlp_adapter llava_checkpoints/llava_stage1_dreamclip30m_NV2dinoL_sequence/mm_projector.bin \
    --tune_alignment_layer True \
    --unlock_vision_tower False \
    --output_dir ./llava_checkpoints/llava_stage2_dreamclip30m_NV2dinoL_sequence
