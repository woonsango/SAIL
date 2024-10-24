#!/bin/bash
#SBATCH --job-name=MMVET
#SBATCH --output=/home/mila/q/qian.yang/LongVLM/Light_Align/evaluation/VLM_eval_scripts/MMVET_output.txt
#SBATCH --error=/home/mila/q/qian.yang/LongVLM/Light_Align/evaluation/VLM_eval_scripts/MMVET_error.txt
#SBATCH --ntasks=1

module load cuda/12.1.1
export PATH="/home/mila/q/qian.yang/anaconda3/envs/FUYU/bin:$PATH"
conda activate FUYU


model_path='/network/scratch/q/qian.yang/light_align/llava_stage1_star7XL_d1024_scale20_sequence_align'
answer_name='llava_stage1_star7XL_d1024_scale20_sequence_align'

python /home/mila/q/qian.yang/LongVLM/Light_Align/VLM_Training/eval/model_vqa_loader.py \
    --model-path $model_path \
    --question-file /home/mila/q/qian.yang/LongVLM/Light_Align/VLM_Training/eval/mm-vet/llava-mm-vet.jsonl \
    --image-folder /home/mila/q/qian.yang/scratch/mmvet/mm-vet/images \
    --answers-file $model_path/mmvet/$answer_name.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

mkdir -p $model_path/mmvet/results

python /home/mila/q/qian.yang/LongVLM/Light_Align/VLM_Training/eval/convert_mmvet_for_eval.py \
    --src $model_path/mmvet/$answer_name.jsonl \
    --dst $model_path/mmvet/results/$answer_name.json

python /home/mila/q/qian.yang/LongVLM/Light_Align/VLM_Training/eval/mm-vet_evaluator.py \
    --mmvet_path /home/mila/q/qian.yang/scratch/mmvet/mm-vet \
    --result_file $model_path/mmvet/results/$answer_name.json \
    --result_path $model_path/mmvet/results