#!/bin/bash
#SBATCH --job-name=nv_12m
#SBATCH --partition=long                       # Ask for unkillable job
#SBATCH --cpus-per-task=4                         # Ask for 8 CPUs
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100l:1                         # Request 2 GPUs
#SBATCH --ntasks-per-node=1                        # Ask for 1 task per node
#SBATCH --mem=64G           
#SBATCH --time=24:00:00                                    
#SBATCH --output=./slurm_logs/encode/%(%Y-%m-%d)T/%x_%j_%A_%a_${data}.out
#SBATCH --error=./slurm_logs/encode/%(%Y-%m-%d)T/%x_%j_%A_%a_${data}.err 





module load miniconda/3
conda init
conda activate aro
module load cudatoolkit/12.1.1

vision_model="facebook/dinov2-giant"
# vision_model="mae-base"
# vision_model="convnextv2-huge"
# vision_model="dinov1-vitb16"
# vision_model="aim_1B"
# vision_model="ibot-base"
# vision_model="r152_2x_sk1"
# vision_model="r101_2x_sk1"

# text_model="Alibaba-NLP/gte-base-en-v1.5"
text_model="nvidia/NV-Embed-v2"
# text_model="Alibaba-NLP/gte-Qwen2-1.5B-instruct"
# data="dreamclipcc12mhf"
# data="dreamclipcc3m"
data="yfcc15m"
domain="text"
batch_size=64
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
    CUDA_VISIBLE_DEVICES=0 python encode.py --domain $domain --vision_model_name $vision_model --text_model_name $text_model --batch_size $batch_size --data $data --resume --end_index 6144000 --source_caption $source_caption &
    # 启动第二个任务在GPU 1上   
    CUDA_VISIBLE_DEVICES=1 python encode.py --domain $domain --vision_model_name $vision_model --text_model_name $text_model --batch_size $batch_size --data $data --resume --start_index 6144000 --end_index 12288000 --source_caption $source_caption &
    # 启动第二个任务在GPU 1上
    CUDA_VISIBLE_DEVICES=2 python encode.py --domain $domain --vision_model_name $vision_model --text_model_name $text_model --batch_size $batch_size --data $data --resume --start_index 12288000 --end_index 18432000 --source_caption $source_caption &
    # 启动第二个任务在GPU 1上
    CUDA_VISIBLE_DEVICES=3 python encode.py --domain $domain --vision_model_name $vision_model --text_model_name $text_model --batch_size $batch_size --data $data --resume --start_index 18432000 --source_caption $source_caption &

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
