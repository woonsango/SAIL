#!/bin/bash
#SBATCH --job-name=knn
#SBATCH --partition=long                         # Ask for unkillable job
#SBATCH --cpus-per-task=12                             # Ask for 2 CPUs
#SBATCH --nodes=1
#SBATCH --gres=gpu:l40s:4
#SBATCH --ntasks-per-node=1                                  # Ask for 1 GPU
#SBATCH --mem=128G           
#SBATCH --time=12:00:00                                    
#SBATCH --output=./slurm_logs/eval/output-%j.txt
#SBATCH --error=./slurm_logs/eval/error-%j.txt 


module load miniconda/3
conda init
conda activate openflamingo

python -m torch.distributed.launch --nproc_per_node=4  eval_knn.py --data_path /home/mila/l/le.zhang/scratch/datasets/imagenet