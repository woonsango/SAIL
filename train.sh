#!/bin/bash
#SBATCH --job-name=light_align_siglip
#SBATCH --partition=long                       # Ask for unkillable job
#SBATCH --cpus-per-task=8                            # Ask for 2 CPUs
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100l:1
#SBATCH --ntasks-per-node=1                                  # Ask for 1 GPU
#SBATCH --mem=256G           
#SBATCH --time=64:00:00                                    
#SBATCH --output=./slurm_logs/sdgen/siglipmlp_o-%j.txt
#SBATCH --error=./slurm_logs/sdgen/siglipmlp_e-%j.txt 

python train.py --text_embedding_dir /home/mila/l/le.zhang/scratch/light_align/data/text_embedding/all-mpnet-base-v2 --image_embedding_dir /home/mila/l/le.zhang/scratch/light_align/data/image_embedding/dinov2-base --batch_size 1024 --output_name siglip