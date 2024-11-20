#!/bin/bash
#SBATCH --job-name=gqa
#SBATCH --partition=unkillable     # Ask for unkillable job
#SBATCH --cpus-per-task=4                             # Ask for 2 CPUs
#SBATCH --nodes=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --ntasks-per-node=1                                  # Ask for 1 GPU
#SBATCH --mem=32G           
#SBATCH --time=1:00:00                                    
#SBATCH --output=./slurm_logs/eval/llava/%x_%j.out
#SBATCH --error=./slurm_logs/eval/llava/%x_%j.err 

module load miniconda/3
conda init
conda activate llava
module load cudatoolkit/12.1.1

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"
# print the gpu list
echo "GPU list: ${GPULIST[@]}"
CHUNKS=${#GPULIST[@]}

model_path='llava_checkpoints/llava_stage2_dreamclip30m_NV2dinoL_sequence_tune_all'
# model_path=$1
SPLIT="llava_gqa_testdev_balanced"
GQADIR="/network/scratch/q/qian.yang/gqa"

IDX=0
python llava_train/eval/model_vqa_loader.py \
    --model-path $model_path \
    --question-file llava_train/eval/gqa/$SPLIT.jsonl \
    --image-folder $GQADIR/images \
    --answers-file $model_path/gqa/$SPLIT/${CHUNKS}_${IDX}.jsonl \
    --num-chunks $CHUNKS \
    --temperature 0 \
    --conv-mode vicuna_v1


output_file=$model_path/gqa/$SPLIT/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

for IDX in $(seq 0 $((CHUNKS-1))); do
    echo "Processing chunk $IDX"
    cat $model_path/gqa/$SPLIT/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

python llava_train/eval/convert_gqa_for_eval.py --src $output_file --dst $GQADIR/testdev_balanced_predictions.json

cd $GQADIR
python eval.py --tier testdev_balanced
