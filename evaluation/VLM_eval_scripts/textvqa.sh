#!/bin/bash
#SBATCH --job-name=TEXTVQA
#SBATCH --output=/home/mila/q/qian.yang/LongVLM/Light_Align/evaluation/VLM_eval_scripts/TEXTVQA_output.txt
#SBATCH --error=/home/mila/q/qian.yang/LongVLM/Light_Align/evaluation/VLM_eval_scripts/TEXTVQA_error.txt
#SBATCH --ntasks=1

module load cuda/12.1.1
export PATH="/home/mila/q/qian.yang/anaconda3/envs/FUYU/bin:$PATH"
conda activate FUYU

model_path='/network/scratch/q/qian.yang/light_align/llava_stage2_star7XL_d1024_scale20_sequence_align'
answer_name='llava_stage2_star7XL_d1024_scale20_sequence_align'

python /home/mila/q/qian.yang/LongVLM/Light_Align/VLM_Training/eval/model_vqa_loader.py \
    --model-path $model_path \
    --question-file /home/mila/q/qian.yang/LongVLM/Light_Align/VLM_Training/eval/textvqa/llava_textvqa_val_v051_ocr.jsonl \
    --image-folder /home/mila/q/qian.yang/scratch/TextVQA/train_images \
    --answers-file $model_path/textvqa/$answer_name.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

python /home/mila/q/qian.yang/LongVLM/Light_Align/VLM_Training/eval/eval_textvqa.py \
    --annotation-file /home/mila/q/qian.yang/scratch/TextVQA/TextVQA_0.5.1_val.json \
    --result-file $model_path/textvqa/$answer_name.jsonl