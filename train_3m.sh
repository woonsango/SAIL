#!/bin/bash
#SBATCH --job-name=cc3mrandfrom12mraw1
#SBATCH --partition=short-unkillable          # Ask for unkillable job
#SBATCH --cpus-per-task=24                             # Ask for 2 CPUs
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100l:1
#SBATCH --ntasks-per-node=1                                  # Ask for 1 GPU
#SBATCH --mem=128G           
#SBATCH --time=3:00:00                                    
#SBATCH --output=./slurm_logs/train/%(%Y-%m-%d)T/%x_%j_%A_%a_${data}.out
#SBATCH --error=./slurm_logs/train/%(%Y-%m-%d)T/%x_%j_%A_%a_${data}.err 

module load miniconda/3
conda init
conda activate openflamingo

# ----------------------TRAIN SETTING------------------------

epoch_num=60
lr=1e-5
bs=32768
# bs=60000
d=1024

text_embedding_list="/home/mila/l/le.zhang/scratch/light_align/data/tensor_data/text_embedding/gte-large-en-v1.5/dreamclipcc12mhf_raw_caption"
image_embedding_list="/home/mila/l/le.zhang/scratch/light_align/data/tensor_data/image_embedding/dinov2-large/dreamclipcc12mhf"
# output_name="cc3mraw_Qwen1.5bdinoG_bs_${bs}_lion_mean_lr_${lr}_star7_d${d}_scale10_negbias10_gmp512"

# text_embedding_list="/home/mila/l/le.zhang/scratch/light_align/data/text_embedding/gte-Qwen2-7B-instruct/dreamclipcc3m_raw_caption"
# image_embedding_list="/home/mila/l/le.zhang/scratch/light_align/data/image_embedding/dinov2-large/dreamclipcc3m"
output_name="cc3mfrom12mraw_gtendinoL_bs_${bs}_lion_mean_lr_${lr}_star7_d${d}_scale10_negbias10"
# ------------------------------------------------------------



timestamp=$(date +%Y%m%d%H%M%S)
job_name=$SLURM_JOB_NAME
export NCCL_DEBUG=INFO

# 检查GPU数量
if [[ "$SLURM_GPUS_ON_NODE" -gt 1 ]]; then
    output_name="${job_name}"
    echo "Running on multiple GPUs"
    export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
    echo master port is $MASTER_POR
    export WORLD_SIZE=$SLURM_NTASKS_PER_NODE
    echo world size is $WORLD_SIZE
    master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
    export MASTER_ADDR=$master_addr
    echo master addr is $MASTER_ADDR
    export OMP_NUM_THREADS=12

    torchrun --master_port $MASTER_PORT --nproc_per_node=$SLURM_GPUS_ON_NODE main.py \
             --text-embedding-list $text_embedding_list \
             --image-embedding-list $image_embedding_list \
             --seed 42 \
             --resume latest \
             --save-frequency 1 \
             --report-to wandb \
             --warmup 50 \
             --batch-size $bs \
             --lr $lr \
             --wd 0.1 \
             --epochs $epoch_num \
             --workers 0 \
             --beta1 0.9 \
             --beta2 0.98 \
             --eps 1e-06 \
             --log-every-n-steps 10 \
             --wandb-project-name clip_training \
             --name $output_name
else
    echo "Running on a single GPU"
    # Be sure to name the output folder with the text and vision model name

    python main.py \
        --text-embedding-list $text_embedding_list \
        --image-embedding-list $image_embedding_list \
        --train-num-samples 2000000 \
        --dataset-type embedding \
        --siglip \
        --seed 42 \
        --resume latest \
        --save-frequency 5 \
        --report-to wandb \
        --batch-size $bs \
        --lr $lr \
        --epochs $epoch_num \
        --workers 0 \
        --wd 1e-07 \
        --target-dimension $d \
        --linear-type star \
        --diagonal-weight 0 \
        --log-every-n-steps 10 \
        --wandb-project-name clip_training \
        --name $output_name \
        --logit_scale 10 \
        --logit_bias -10 \
        --gmp_groups 512 \
        --use_gmp

fi

if [ $? -ne 0 ]; then
    echo "Training failed. Checking for checkpoints..."
    
    if ls ./logs/$output_name/checkpoints/*.pt 1> /dev/null 2>&1; then
        echo "Checkpoint file(s) found. Keeping the log directory."
    else
        echo "No checkpoint files found. Cleaning up ./logs/${output_name}"
        rm -rf ./logs/$output_name
    fi
fi
