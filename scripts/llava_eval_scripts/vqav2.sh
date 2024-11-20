#!/bin/bash
#SBATCH --job-name=vqav2
#SBATCH --partition=short-unkillable     # Ask for unkillable job
#SBATCH --cpus-per-task=4                             # Ask for 2 CPUs
#SBATCH --nodes=1
#SBATCH --gres=gpu:l40s:4
#SBATCH --ntasks-per-node=1                                  # Ask for 1 GPU
#SBATCH --mem=32G           
#SBATCH --time=3:00:00                                    
#SBATCH --output=./slurm_logs/eval/%x_%j.out
#SBATCH --error=./slurm_logs/eval/%x_%j.err 

module load miniconda/3
conda init
conda activate llava
module load cudatoolkit/12.1.1

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

QIAN_SCRATCH='/network/scratch/q/qian.yang'
model_path='llava_checkpoints/llava_stage2_dreamclip30m_NV2dinoL_sequence_tune_all'
# model_path=$1
model_path=${model_path##*/}

SPLIT="llava_vqav2_mscoco_test-dev2015"

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python llava_train/eval/model_vqa_loader.py \
        --model-path $model_path \
        --question-file $SCRATCH/light_align/llava_train/eval/vqav2/$SPLIT.jsonl \
        --image-folder $QIAN_SCRATCH/vqav2/test2015 \
        --answers-file $model_path/vqav2/$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --conv-mode vicuna_v1 &
done

wait

output_file=$model_path/vqav2/$SPLIT/$model_path/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat $model_path/vqav2/$SPLIT/$model_path/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

python llava_train/eval/convert_vqav2_for_submission.py --split $SPLIT --model_path $model_path --dir $model_path/vqav2

