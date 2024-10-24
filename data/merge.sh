#!/bin/bash
#SBATCH --job-name=merge
#SBATCH --partition=long-cpu                      # Ask for unkillable job
#SBATCH --cpus-per-task=4                             # Ask for 2 CPUs
#SBATCH --mem=128G           
#SBATCH --time=9:00:00                                    
#SBATCH --output=./slurm_logs/train/output-%j.txt
#SBATCH --error=./slurm_logs/train/error-%j.txt 

module load miniconda/3
conda init
conda activate openflamingo

python create_h5.py