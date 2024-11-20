#!/bin/bash
#SBATCH --job-name=seed
#SBATCH --output=/home/mila/q/qian.yang/LongVLM/Light_Align/evaluation/VLM_eval_scripts/seed_output.txt
#SBATCH --error=/home/mila/q/qian.yang/LongVLM/Light_Align/evaluation/VLM_eval_scripts/seed_error.txt
#SBATCH --ntasks=1


module load miniconda/3
conda init
conda activate llava
module load cudatoolkit/12.1.1

export PYTHONPATH=/network/scratch/l/le.zhang/light_align:$PYTHONPATH

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

QIAN_SCRATCH='/network/scratch/q/qian.yang'
CHUNKS=${#GPULIST[@]}
# model_path='llava_checkpoints/llava_stage2_dreamclip30m_NV2dinoL_sequence_tune_all'
model_path=$1
CKPT=${model_path##*/}

# IDX=0
# python llava_train/eval/model_vqa_loader.py \
#     --model-path $model_path \
#     --question-file llava_train/eval/seed_bench/llava-seed-bench.jsonl \
#     --image-folder $QIAN_SCRATCH/seedbench \
#     --answers-file $model_path/seed/answers/$model_path/${CHUNKS}_${IDX}.jsonl \
#     --num-chunks $CHUNKS \
#     --chunk-idx $IDX \
#     --temperature 0 \
#     --conv-mode vicuna_v1


output_file=$model_path/seed/answers/$CKPT/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat $model_path/seed/answers/$CKPT/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

# Evaluate
python llava_train/eval/convert_seed_for_submission.py \
    --annotation-file $QIAN_SCRATCH/seedbench/SEED-Bench.json \
    --result-file $output_file \
    --result-upload-file $model_path/seed/$CKPT.jsonl
