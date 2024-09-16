#!/bin/bash
#SBATCH --job-name=dinog12m
#SBATCH --partition=short-unkillable                           # Ask for unkillable job
#SBATCH --cpus-per-task=24                         # Ask for 8 CPUs
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100l:4                         # Request 2 GPUs
#SBATCH --ntasks-per-node=1                        # Ask for 1 task per node
#SBATCH --mem=128G           
#SBATCH --time=3:00:00                                    
#SBATCH --output=./slurm_logs/encode/%(%Y-%m-%d)T/%x_%j_%A_%a_${data}.out
#SBATCH --error=./slurm_logs/encode/%(%Y-%m-%d)T/%x_%j_%A_%a_${data}.err 





module load miniconda/3
conda init
conda activate openflamingo
module load cudatoolkit/12.1.1

vision_model="facebook/dinov2-giant"
# vision_model="facebook/vit-mae-large"
text_model="Alibaba-NLP/gte-large-en-v1.5"
# text_model="Alibaba-NLP/gte-Qwen2-1.5B-instruct"
data="dreamclipcc12mhf"
# data="dreamclipcc3m"
domain="image"
batch_size=4096
# select one of the following source captions
# raw_caption longIB_captions longSV_captions longLLA_captions | shortIB_captions shortSV_captions shortLLA_captions
source_caption="raw_caption"
# Program 
gpu_count=$SLURM_GPUS_ON_NODE

if [ "$gpu_count" -eq 4 ]; then
    echo "Running tasks in parallel on multiple GPUs..."
    echo "Using vision model: $vision_model"
    echo "Using text model: $text_model"
    echo "Processing dataset: $data"
    echo "Using domain: $domain"
    echo "Each GPU will use save batch size of $batch_size"
    echo "Using source caption: $source_caption"
    # 启动第一个任务在GPU 0上
    CUDA_VISIBLE_DEVICES=0 python encode.py --domain $domain --vision_model_name $vision_model --text_model_name $text_model --batch_size $batch_size --data $data --resume --end_index 2000000 --source_caption $source_caption &
    # 启动第二个任务在GPU 1上   
    CUDA_VISIBLE_DEVICES=1 python encode.py --domain $domain --vision_model_name $vision_model --text_model_name $text_model --batch_size $batch_size --data $data --resume --start_index 2000000 --end_index 4000000 --source_caption $source_caption &
    # 启动第二个任务在GPU 1上
    CUDA_VISIBLE_DEVICES=2 python encode.py --domain $domain --vision_model_name $vision_model --text_model_name $text_model --batch_size $batch_size --data $data --resume --start_index 4000000 --end_index 6000000 --source_caption $source_caption &
    # 启动第二个任务在GPU 1上
    CUDA_VISIBLE_DEVICES=3 python encode.py --domain $domain --vision_model_name $vision_model --text_model_name $text_model --batch_size $batch_size --data $data --resume --start_index 6000000 --source_caption $source_caption &

    # 等待所有后台任务完成
    wait
else
    echo "Running tasks sequentially on a single GPU..."
    echo "Using vision model: $vision_model"
    echo "Using text model: $text_model"
    echo "Processing dataset: $data"
    echo "Using domain: $domain"
    echo "Using batch size: $batch_size"
    echo "Using source caption: $source_caption"
    CUDA_VISIBLE_DEVICES=0 python encode.py --domain $domain --vision_model_name $vision_model --text_model_name $text_model --batch_size $batch_size --data $data --resume --source_caption $source_caption 
fi