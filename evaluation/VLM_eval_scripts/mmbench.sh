#!/bin/bash
#SBATCH --job-name=MMBench
#SBATCH --output=/home/mila/q/qian.yang/LongVLM/Light_Align/evaluation/VLM_eval_scripts/MMBench_output.txt
#SBATCH --error=/home/mila/q/qian.yang/LongVLM/Light_Align/evaluation/VLM_eval_scripts/MMBench_error.txt
#SBATCH --ntasks=1

module load cuda/12.1.1
export PATH="/home/mila/q/qian.yang/anaconda3/envs/FUYU/bin:$PATH"
conda activate FUYU

SPLIT="mmbench_dev_20230712"
model_path='/network/scratch/q/qian.yang/light_align/llava_stage2_star7XL_d1024_scale20_sequence_align'
answer_name='llava_stage2_star7XL_d1024_scale20_sequence_align'

python /home/mila/q/qian.yang/LongVLM/Light_Align/VLM_Training/eval/model_vqa_mmbench.py \
    --model-path $model_path \
    --question-file /home/mila/q/qian.yang/scratch/mmbench/$SPLIT.tsv \
    --answers-file $model_path/mmbench/$SPLIT/$answer_name.jsonl \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode vicuna_v1

mkdir -p $model_path/mmbench/answers_upload/$SPLIT

python /home/mila/q/qian.yang/LongVLM/Light_Align/VLM_Training/eval/convert_mmbench_for_submission.py \
    --annotation-file /home/mila/q/qian.yang/scratch/mmbench/$SPLIT.tsv \
    --result-dir $model_path/mmbench/$SPLIT \
    --upload-dir $model_path/mmbench/answers_upload/$SPLIT \
    --experiment $answer_name