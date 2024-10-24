#!/bin/bash
#SBATCH --job-name=vqav2
#SBATCH --output=/home/mila/q/qian.yang/LongVLM/Light_Align/evaluation/VLM_eval_scripts/vqav2_output.txt
#SBATCH --error=/home/mila/q/qian.yang/LongVLM/Light_Align/evaluation/VLM_eval_scripts/vqav2_error.txt
#SBATCH --ntasks=1

module load cuda/12.1.1
export PATH="/home/mila/q/qian.yang/anaconda3/envs/FUYU/bin:$PATH"
conda activate FUYU
gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

model_path='/network/scratch/q/qian.yang/light_align/llava_stage2_star7XL_d1024_scale20_sequence_align'
CKPT='llava_stage2_star7XL_d1024_scale20_sequence_align'

SPLIT="llava_vqav2_mscoco_test-dev2015"

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python /home/mila/q/qian.yang/LongVLM/Light_Align/VLM_Training/eval/model_vqa_loader.py \
        --model-path $model_path \
        --question-file /home/mila/q/qian.yang/LongVLM/Light_Align/VLM_Training/eval/vqav2/$SPLIT.jsonl \
        --image-folder /home/mila/q/qian.yang/scratch/vqav2/test2015 \
        --answers-file $model_path/vqav2/$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --conv-mode vicuna_v1 &
done

wait

output_file=$model_path/vqav2/$SPLIT/$CKPT/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat $model_path/vqav2/$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

python /home/mila/q/qian.yang/LongVLM/Light_Align/VLM_Training/eval/convert_vqav2_for_submission.py --split $SPLIT --ckpt $CKPT --dir $model_path/vqav2

