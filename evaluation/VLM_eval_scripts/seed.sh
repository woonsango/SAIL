#!/bin/bash
#SBATCH --job-name=seed
#SBATCH --output=/home/mila/q/qian.yang/LongVLM/Light_Align/evaluation/VLM_eval_scripts/seed_output.txt
#SBATCH --error=/home/mila/q/qian.yang/LongVLM/Light_Align/evaluation/VLM_eval_scripts/seed_error.txt
#SBATCH --ntasks=1

module load cuda/12.1.1
export PATH="/home/mila/q/qian.yang/anaconda3/envs/FUYU/bin:$PATH"
conda activate FUYU

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

model_path='/network/scratch/q/qian.yang/light_align/llava_stage2_star7XL_d1024_scale20_sequence_align'
CKPT='llava_stage2_star7XL_d1024_scale20_sequence_align'

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python /home/mila/q/qian.yang/LongVLM/Light_Align/VLM_Training/eval/model_vqa_loader.py \
    --model-path $model_path \
        --question-file /home/mila/q/qian.yang/LongVLM/Light_Align/VLM_Training/eval/seed_bench/llava-seed-bench.jsonl \
        --image-folder /home/mila/q/qian.yang/scratch/seedbench \
        --answers-file $model_path/seed/answers/$CKPT/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --conv-mode vicuna_v1 &
done

wait

output_file=$model_path/seed/answers/$CKPT/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat $model_path/seed/answers/$CKPT/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

# Evaluate
python /home/mila/q/qian.yang/LongVLM/Light_Align/VLM_Training/eval/convert_seed_for_submission.py \
    --annotation-file /home/mila/q/qian.yang/scratch/seedbench/SEED-Bench.json \
    --result-file $output_file \
    --result-upload-file $model_path/seed/$CKPT.jsonl

