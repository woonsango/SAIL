#!/bin/bash
#SBATCH --job-name=Vizwiz
#SBATCH --output=/home/mila/q/qian.yang/LongVLM/Light_Align/evaluation/VLM_eval_scripts/Vizwiz_output.txt
#SBATCH --error=/home/mila/q/qian.yang/LongVLM/Light_Align/evaluation/VLM_eval_scripts/Vizwiz_error.txt
#SBATCH --ntasks=1

module load cuda/12.1.1
export PATH="/home/mila/q/qian.yang/anaconda3/envs/FUYU/bin:$PATH"
conda activate FUYU


model_path='/network/scratch/q/qian.yang/light_align/llava_stage2_star7XL_d1024_scale20_sequence_align'
answer_name='llava_stage2_star7XL_d1024_scale20_sequence_align'
upload_name='vizwiz_answer_upload_'$answer_name


python /home/mila/q/qian.yang/LongVLM/Light_Align/VLM_Training/eval/model_vqa_loader.py \
    --model-path $model_path \
    --question-file /home/mila/q/qian.yang/LongVLM/Light_Align/VLM_Training/eval/vizwiz/llava_test.jsonl \
    --image-folder /home/mila/q/qian.yang/scratch/vizwiz/test \
    --answers-file $model_path/vizwiz/$answer_name.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

python /home/mila/q/qian.yang/LongVLM/Light_Align/VLM_Training/eval/convert_vizwiz_for_submission.py \
    --annotation-file /home/mila/q/qian.yang/LongVLM/Light_Align/VLM_Training/eval/vizwiz/llava_test.jsonl \
    --result-file $model_path/vizwiz/$answer_name.jsonl \
    --result-upload-file $model_path/vizwiz/$upload_name.json