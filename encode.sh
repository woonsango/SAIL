#!/bin/bash
#SBATCH --job-name=encode_sharegpt
#SBATCH --partition=long                           # Ask for unkillable job
#SBATCH --cpus-per-task=4                             # Ask for 2 CPUs
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100l:1
#SBATCH --ntasks-per-node=1                                  # Ask for 1 GPU
#SBATCH --mem=128G           
#SBATCH --time=24:00:00                                    
#SBATCH --output=./slurm_logs/encode/output-%j.txt
#SBATCH --error=./slurm_logs/encode/error-%j.txt 


module load miniconda/3
conda init
conda activate openflamingo

# vision_model="facebook/dinov2-base"
# model="Alibaba-NLP/gte-Qwen2-1.5B-instruct"
vision_model="facebook/dinov2-large"
text_model="Alibaba-NLP/gte-large-en-v1.5"
data="Sharegpt4vllava"
# data="ALLaVAVFLAN"

python encode.py --domain text --model $text_model --batch_size 512 --data $data --resume
python encode.py --domain image --model $vision_model  --batch_size 1024 --data $data --resume

