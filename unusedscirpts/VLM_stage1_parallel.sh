#!/bin/bash
#SBATCH --job-name=llava1.5_pretrain
#SBATCH --partition=short-unkillable       # Ask for unkillable job
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
module load cudatoolkit/12.1.1


deepspeed --num_gpus=4 VLM_Training/train_mem.py \
    --deepspeed VLM_Training/zero2.json \
    --model_name_or_path lmsys/vicuna-7b-v1.5 \
    --version plain \
    --data_path /network/scratch/q/qian.yang/llava-v1.5-7b/pretrain_data/blip_laion_cc_sbu_558k.json \
    --target_dimension 1024 \
    --linear_type star \
    --image_folder /network/scratch/q/qian.yang/llava-v1.5-7b/pretrain_data \
    --vision_tower facebook/dinov2-large \
    --mm_projector_type mlp2x_gelu \
    --tune_mm_mlp_adapter True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --num_train_epochs 1 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 4000 \
    --save_total_limit 1 \
    --learning_rate 0.001 \
    --weight_decay 0.0 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --fp16 False \
    --bf16 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4  \
    --lazy_preprocess True \
    --report_to wandb  \
    --vlhead_weights_path logs/dreamclip30m_gtendinoL_bs_32768_lion_mean_lr_1e-5_star7XL_d1024_scale20_bias-10_multi_postext_s2/checkpoints/epoch_30.pt \
    --parallel_enable \
    --output_dir ./llava_checkpoints/llava_stage1_star7XL_d1024_scale20_parallel
