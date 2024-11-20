#!/bin/bash
#SBATCH --job-name=Pope
#SBATCH --output=/home/mila/q/qian.yang/LongVLM/Light_Align/evaluation/VLM_eval_scripts/Pope_output.txt
#SBATCH --error=/home/mila/q/qian.yang/LongVLM/Light_Align/evaluation/VLM_eval_scripts/Pope_error.txt
#SBATCH --ntasks=1


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
    --question-file $SCRATCH/light_align/llava_train/eval/pope/llava_pope_test.jsonl \
    --image-folder $QIAN_SCRATCH/coco2014/val2014 \
    --answers-file $model_path/pope/$SPLIT/$answer_name.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

python llava_train/eval/eval_pope.py \
    --annotation-dir $SCRATCH/light_align/llava_train/eval/pope \
    --question-file $SCRATCH/light_align/llava_train/eval/pope/llava_pope_test.jsonl \
    --result-file $model_path/pope/$SPLIT/$answer_name.jsonl