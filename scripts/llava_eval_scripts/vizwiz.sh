#!/bin/bash
#SBATCH --job-name=vizwiz
#SBATCH --partition=long     # Ask for unkillable job
#SBATCH --cpus-per-task=4                             # Ask for 2 CPUs
#SBATCH --nodes=1
#SBATCH --gres=gpu:l40s:4
#SBATCH --ntasks-per-node=1                                  # Ask for 1 GPU
#SBATCH --mem=32G           
#SBATCH --time=1:00:00                                    
#SBATCH --output=./slurm_logs/eval/%x_%j.out
#SBATCH --error=./slurm_logs/eval/%x_%j.err 
#SBATCH --ntasks=1

module load miniconda/3
conda init
conda activate llava
module load cudatoolkit/12.1.1


QIAN_SCRATCH='/network/scratch/q/qian.yang'

model_path='llava_checkpoints/llava_stage2_dreamclip30m_NV2dinoL_sequence_tune_all'
# model_path=$1
answer_name=${model_path##*/}
upload_name='vizwiz_answer_upload_'$answer_name


python llava_train/eval/model_vqa_loader.py \
    --model-path $model_path \
    --question-file $SCRATCH/light_align/llava_train/eval/vizwiz/llava_test.jsonl \
    --image-folder $QIAN_SCRATCH/vizwiz/test \
    --answers-file $model_path/vizwiz/$answer_name.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

python llava_train/eval/convert_vizwiz_for_submission.py \
    --annotation-file $SCRATCH/light_align/llava_train/eval/vizwiz/llava_test.jsonl \
    --result-file $model_path/vizwiz/$answer_name.jsonl \
    --result-upload-file $model_path/vizwiz/$upload_name.json