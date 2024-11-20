#!/bin/bash
#SBATCH --job-name=Stage2_Parallel
#SBATCH --output=VLM_stage2_parallel_output.txt
#SBATCH --error=VLM_stage2_parallel_error.txt
#SBATCH --ntasks=1

module load cuda/12.1.1
#!/bin/bash
export PATH="/home/mila/q/qian.yang/anaconda3/envs/FUYU/bin:$PATH"
conda activate FUYU

deepspeed --num_gpus=4 VLM_Training/train_mem.py \
    --deepspeed VLM_Training/zero3.json \
    --model_name_or_path /network/scratch/q/qian.yang/vicuna-7b-v1.5 \
    --version v1 \
    --data_path /network/scratch/q/qian.yang/llava-v1.5-7b/instruct_tuning_data/llava_v1_5_mix665k.json \
    --image_folder /network/scratch/q/qian.yang/llava-v1.5-7b/instruct_tuning_data/data \
    --target_dimension 1024 \
    --linear_type star \
    --vlhead_weights_path /network/scratch/l/le.zhang/light_align/logs/dreamclip30m_gtendinoL_bs_32768_lion_mean_lr_1e-5_star7XL_d1024_scale20_bias-10_multi_postext_s2/checkpoints/epoch_30.pt \
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
    --save_steps 900 \
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
    --pretrain_mm_mlp_adapter /network/scratch/q/qian.yang/light_align/llava_stage1_star7XL_d1024_scale20_parallel/mm_projector.bin \
    --parallel_enable \
    --pretrain_mm_mlp_down_projector /network/scratch/q/qian.yang/light_align/llava_stage1_star7XL_d1024_scale20_parallel/down_projector.bin \
    --output_dir /network/scratch/q/qian.yang/light_align/llava_stage2_star7XL_d1024_scale20_parallel