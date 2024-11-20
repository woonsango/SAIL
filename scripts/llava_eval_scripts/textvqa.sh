#!/bin/bash
#SBATCH --job-name=textvqa
#SBATCH --partition=unkillable     # Ask for unkillable job
#SBATCH --cpus-per-task=4                             # Ask for 2 CPUs
#SBATCH --nodes=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --ntasks-per-node=1                                  # Ask for 1 GPU
#SBATCH --mem=32G           
#SBATCH --time=1:00:00                                    
#SBATCH --output=./slurm_logs/eval/llava/%x_%j.out
#SBATCH --error=./slurm_logs/eval/llava/%x_%j.err 

module load miniconda/3
conda init
conda activate llava
module load cudatoolkit/12.1.1

QIAN_SCRATCH='/network/scratch/q/qian.yang'

model_path='llava_checkpoints/llava_stage2_dreamclip30m_NV2dinoL_sequence_tune_all'
# model_path=$1
answer_name=${model_path##*/}

python llava_train/eval/model_vqa_loader.py \
    --model-path $model_path \
    --question-file $SCRATCH/light_align/llava_train/eval/textvqa/llava_textvqa_val_v051_ocr.jsonl \
    --image-folder $QIAN_SCRATCH/TextVQA/train_images \
    --answers-file $model_path/textvqa/$answer_name.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

python llava_train/eval/eval_textvqa.py \
    --annotation-file $QIAN_SCRATCH/TextVQA/TextVQA_0.5.1_val.json \
    --result-file $model_path/textvqa/$answer_name.jsonl