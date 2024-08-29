#!/bin/bash
#SBATCH --job-name=merge
#SBATCH --partition=long-cpu                      # Ask for unkillable job
#SBATCH --cpus-per-task=4                             # Ask for 2 CPUs
#SBATCH --mem=256G           
#SBATCH --time=3:00:00                                    
#SBATCH --output=./slurm_logs/train/output-%j.txt
#SBATCH --error=./slurm_logs/train/error-%j.txt 

module load miniconda/3
conda init
conda activate openflamingo

python /home/mila/l/le.zhang/scratch/light_align/data/merge_file.py