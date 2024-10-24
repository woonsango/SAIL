#!/bin/bash
#SBATCH --job-name=Stage1_seq_align  
#SBATCH --output=/home/mila/q/qian.yang/LongVLM/Light_Align/VLM_stage1_sequence_align_output.txt
#SBATCH --error=/home/mila/q/qian.yang/LongVLM/Light_Align/VLM_stage1_sequence_align_error.txt
#SBATCH --ntasks=1

module load cuda/12.1.1
#!/bin/bash
export PATH="/home/mila/q/qian.yang/anaconda3/envs/FUYU/bin:$PATH"
conda activate FUYU

deepspeed --num_gpus=4 /home/mila/q/qian.yang/LongVLM/Light_Align/VLM_Training/train_mem.py \
    --deepspeed /home/mila/q/qian.yang/LongVLM/Light_Align/VLM_Training/zero2.json \
    --model_name_or_path /network/scratch/q/qian.yang/vicuna-7b-v1.5 \
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
    --vlhead_weights_path /network/scratch/l/le.zhang/light_align/logs/dreamclip30m_gtendinoL_bs_32768_lion_mean_lr_1e-5_star7XL_d1024_scale20_bias-10_multi_postext_s2/checkpoints/epoch_30.pt \
    --alignment_enable True \
    --output_dir /network/scratch/q/qian.yang/light_align/llava_stage1_star7XL_d1024_scale20_sequence_align