#!/bin/bash
#SBATCH --job-name=MMBench
#SBATCH --output=/home/mila/q/qian.yang/LongVLM/Light_Align/evaluation/VLM_eval_scripts/MMBench_output.txt
#SBATCH --error=/home/mila/q/qian.yang/LongVLM/Light_Align/evaluation/VLM_eval_scripts/MMBench_error.txt
#SBATCH --ntasks=1

module load miniconda/3
conda init
conda activate llava
module load cudatoolkit/12.1.1

QIAN_SCRATCH='/network/scratch/q/qian.yang'
SPLIT="mmbench_dev_20230712"


# model_path='llava_checkpoints/llava_stage2_dreamclip30m_NV2dinoL_sequence_freeze_vlhead'
model_path=$1
answer_name=${model_path##*/}

python llava_train/eval/model_vqa_mmbench.py \
    --model-path $model_path \
    --question-file $QIAN_SCRATCH/mmbench/$SPLIT.tsv \
    --answers-file $model_path/mmbench/$SPLIT/$answer_name.jsonl \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode vicuna_v1

mkdir -p $model_path/mmbench/answers_upload/$SPLIT

python llava_train/eval/convert_mmbench_for_submission.py \
    --annotation-file $QIAN_SCRATCH/mmbench/$SPLIT.tsv \
    --result-dir $model_path/mmbench/$SPLIT \
    --upload-dir $model_path/mmbench/answers_upload/$SPLIT \
    --experiment $answer_name