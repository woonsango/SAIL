#!/bin/bash
#SBATCH --job-name=light_align_siglip
#SBATCH --partition=short-unkillable                       # Ask for unkillable job
#SBATCH --cpus-per-task=24                            # Ask for 2 CPUs
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100l:4
#SBATCH --ntasks-per-node=1                                  # Ask for 1 GPU
#SBATCH --mem=128G           
#SBATCH --time=3:00:00                                    
#SBATCH --output=./slurm_logs/sdgen/siglipmlp_o-%j.txt
#SBATCH --error=./slurm_logs/sdgen/siglipmlp_e-%j.txt 

# torchrun --nproc_per_node=1 train.py --data_path /home/mila/l/le.zhang/scratch/light_align/data/blip_laion_cc_sbu_558k.json --image_dir /home/mila/l/le.zhang/scratch/light_align/data/image --batch_size 8 --output_name raw_data_4gpu --save_n_iter 2 --num_epochs 300

python train.py --data_path /home/mila/l/le.zhang/scratch/light_align/data/blip_laion_cc_sbu_558k.json --image_dir /home/mila/l/le.zhang/scratch/light_align/data/image --batch_size 8 --output_name raw_data_4gpu --save_n_iter 2 --num_epochs 300