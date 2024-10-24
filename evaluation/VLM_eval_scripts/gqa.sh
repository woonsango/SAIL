#!/bin/bash
#SBATCH --job-name=GQA
#SBATCH --output=/home/mila/q/qian.yang/LongVLM/Light_Align/evaluation/VLM_eval_scripts/GQA_output.txt
#SBATCH --error=/home/mila/q/qian.yang/LongVLM/Light_Align/evaluation/VLM_eval_scripts/GQA_error.txt
#SBATCH --ntasks=1

module load cuda/12.1.1
export PATH="/home/mila/q/qian.yang/anaconda3/envs/FUYU/bin:$PATH"
conda activate FUYU

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"
# print the gpu list
echo "GPU list: ${GPULIST[@]}"
CHUNKS=${#GPULIST[@]}

CKPT='/network/scratch/q/qian.yang/light_align/llava_stage2_star7XL_d1024_scale20_sequence_align'
SPLIT="llava_gqa_testdev_balanced"
GQADIR="/home/mila/q/qian.yang/scratch/gqa"

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python /home/mila/q/qian.yang/LongVLM/Light_Align/VLM_Training/eval/model_vqa_loader.py \
        --model-path $CKPT \
        --question-file /home/mila/q/qian.yang/LongVLM/Light_Align/VLM_Training/eval/gqa/$SPLIT.jsonl \
        --image-folder $GQADIR/images \
        --answers-file $CKPT/gqa/$SPLIT/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --conv-mode vicuna_v1 &
done

wait

output_file=$CKPT/gqa/$SPLIT/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

for IDX in $(seq 0 $((CHUNKS-1))); do
    echo "Processing chunk $IDX"
    cat $CKPT/gqa/$SPLIT/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

python /home/mila/q/qian.yang/LongVLM/Light_Align/VLM_Training/eval/convert_gqa_for_eval.py --src $output_file --dst $GQADIR/testdev_balanced_predictions.json

cd $GQADIR
python eval.py --tier testdev_balanced
