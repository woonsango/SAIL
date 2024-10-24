#!/bin/bash
#SBATCH --job-name=Stage2_seq_align
#SBATCH --output=/home/mila/q/qian.yang/LongVLM/Light_Align/VLM_stage2_sequence_align_output.txt
#SBATCH --error=/home/mila/q/qian.yang/LongVLM/Light_Align/VLM_stage2_sequence_align_error.txt
#SBATCH --ntasks=1

module load cuda/12.1.1
#!/bin/bash
export PATH="/home/mila/q/qian.yang/anaconda3/envs/FUYU/bin:$PATH"
conda activate FUYU

deepspeed --num_gpus=4 /home/mila/q/qian.yang/LongVLM/Light_Align/VLM_Training/train_mem.py \
    --deepspeed /home/mila/q/qian.yang/LongVLM/Light_Align/VLM_Training/zero3.json \
    --model_name_or_path /network/scratch/q/qian.yang/vicuna-7b-v1.5 \
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
    --save_steps 600 \
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
    --alignment_enable True \
    --vlhead_weights_path /network/scratch/q/qian.yang/light_align/llava_stage1_star7XL_d1024_scale20_sequence_align/vlhead.bin \
    --pretrain_mm_mlp_adapter /network/scratch/q/qian.yang/light_align/llava_stage1_star7XL_d1024_scale20_sequence_align/mm_projector.bin \
    --output_dir /network/scratch/q/qian.yang/light_align/llava_stage2_star7XL_d1024_scale20_sequence_align