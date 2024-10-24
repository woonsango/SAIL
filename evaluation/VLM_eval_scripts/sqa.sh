#!/bin/bash
#SBATCH --job-name=SQA
#SBATCH --output=/home/mila/q/qian.yang/LongVLM/Light_Align/evaluation/VLM_eval_scripts/SQA_output.txt
#SBATCH --error=/home/mila/q/qian.yang/LongVLM/Light_Align/evaluation/VLM_eval_scripts/SQA_error.txt
#SBATCH --ntasks=1

module load cuda/12.1.1
export PATH="/home/mila/q/qian.yang/anaconda3/envs/FUYU/bin:$PATH"
conda activate FUYU


model_path='/network/scratch/q/qian.yang/light_align/llava_stage2_star7XL_d1024_scale20_sequence_align'
answer_name='llava_stage2_star7XL_d1024_scale20_sequence_align'

python /home/mila/q/qian.yang/LongVLM/Light_Align/VLM_Training/eval/model_vqa_science.py \
    --model-path $model_path \
    --question-file /home/mila/q/qian.yang/LongVLM/Light_Align/VLM_Training/eval/scienceqa/llava_test_CQM-A.json \
    --image-folder /home/mila/q/qian.yang/scratch/ScienceQA/test \
    --answers-file $model_path/scienceqa/$SPLIT/$answer_name.jsonl \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode vicuna_v1

python /home/mila/q/qian.yang/LongVLM/Light_Align/VLM_Training/eval/eval_science_qa.py \
    --base-dir /home/mila/q/qian.yang/scratch/ScienceQA \
    --result-file $model_path/scienceqa/$SPLIT/$answer_name.jsonl \
    --output-file $model_path/scienceqa/$SPLIT/$answer_name.jsonl \
    --output-result $model_path/scienceqa/$SPLIT/$answer_name.jsonl
