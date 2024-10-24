#!/bin/bash
#SBATCH --job-name=eval
#SBATCH --partition=unkillable                           # Ask for unkillable job
#SBATCH --cpus-per-task=4                             # Ask for 2 CPUs
#SBATCH --nodes=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --ntasks-per-node=1                                  # Ask for 1 GPU
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --output=./slurm_logs/eval/output-%j.txt
#SBATCH --error=./slurm_logs/eval/error-%j.txt

module load miniconda/3
conda init
conda activate FUYU
# conda activate openflamingo
export HF_AUTH_TOKEN="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"

# -----------------------MODEL SETTING------------------------
# vision_model="facebook/dinov2-base"
vision_model="facebook/dinov2-large"
# vision_model="facebook/dinov2-giant"
# vision_model="facebook/vit-mae-large"

# text_model="sentence-transformers/all-mpnet-base-v2"
text_model="Alibaba-NLP/gte-large-en-v1.5"
# text_model="Alibaba-NLP/gte-Qwen2-1.5B-instruct"
# ------------------------------------------------------------

head_weights_path="/network/scratch/l/le.zhang/light_align/logs/dreamclip30m_gtendinoL_bs_32768_lion_mean_lr_1e-5_star7XL_d1024_scale20_bias-10_multi_postext_s2/checkpoints/epoch_30.pt"

dataset_root="/network/scratch/q/qian.yang/light_align/datasets"
# the cache folder for the datasets

output_root=/network/scratch/q/qian.yang/light_align/results

if [ ! -d "$output_root" ]; then
    mkdir -p "$output_root"
fi

target_dimension=1024

# cars is not available right now, so we skip it now
# we can get it from https://github.com/pytorch/vision/issues/7545#issuecomment-1631441616

# for caltech101, please download from https://drive.google.com/file/d/137RyRjvTBkBiIfeYBNZBtViDHQ6_Ewsp and https://drive.google.com/file/d/175kQy3UsZ0wUEHZjqkUDdNVssr7bgh_m
# and place them in the dataset_root/caltech101 folder
for dataset in food101 cifar10 cifar100 fgvc_aircraft dtd pets caltech101 flowers; do
    echo "Evaluating on $dataset"
    python cli.py \
        --text_model "$text_model" \
        --vision_model "$vision_model" \
        --head_weights_path "$head_weights_path" \
        --linear_type star \
        --target_dimension "$target_dimension" \
        eval \
        --model "$checkpoint_path" \
        --dataset "$dataset" \
        --dataset_root "$dataset_root" \
        --language en \
        --task auto \
        --output "${output_root}/${dataset}_${vision_model##*/}_${text_model##*/}_en_auto.json" \
        --batch_size 64
done
