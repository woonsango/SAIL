#!/bin/bash

#SBATCH --job-name=Pope
#SBATCH --output=/home/mila/q/qian.yang/LongVLM/Light_Align/evaluation/VLM_eval_scripts/Pope_output.txt
#SBATCH --error=/home/mila/q/qian.yang/LongVLM/Light_Align/evaluation/VLM_eval_scripts/Pope_error.txt
#SBATCH --ntasks=1

module load cuda/12.1.1
export PATH="/home/mila/q/qian.yang/anaconda3/envs/FUYU/bin:$PATH"
conda activate FUYU


model_path='/network/scratch/q/qian.yang/light_align/llava_stage2_star7XL_d1024_scale20_sequence_align'
answer_name='llava_stage2_star7XL_d1024_scale20_sequence_align'

python /home/mila/q/qian.yang/LongVLM/Light_Align/VLM_Training/eval/model_vqa_loader.py \
    --model-path $model_path \
    --question-file /home/mila/q/qian.yang/LongVLM/Light_Align/VLM_Training/eval/pope/llava_pope_test.jsonl \
    --image-folder /home/mila/q/qian.yang/scratch/coco2014/val2014 \
    --answers-file $model_path/pope/$SPLIT/$answer_name.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

python /home/mila/q/qian.yang/LongVLM/Light_Align/VLM_Training/eval/eval_pope.py \
    --annotation-dir /home/mila/q/qian.yang/LongVLM/Light_Align/VLM_Training/eval/pope \
    --question-file /home/mila/q/qian.yang/LongVLM/Light_Align/VLM_Training/eval/pope/llava_pope_test.jsonl \
    --result-file $model_path/pope/$SPLIT/$answer_name.jsonl