import torch
import os
import json
from tqdm import tqdm
import numpy as np
import json
import argparse
from model import SentenceEmbedding, ImageEmbedding
from train.logger import setup_logging
import logging
import pandas as pd
import csv
setup_logging(log_file = None, level = logging.INFO)


DATADIR = {
    'LLaVA558K' : {
        'annotation':'/home/mila/l/le.zhang/scratch/light_align/data/blip_laion_cc_sbu_558k.json',
        'imagedir':'/home/mila/l/le.zhang/scratch/light_align/data/image'
        },
    'ALLaVALAION' : {
        # note that some images are missing in the original repo, thus we filter out invalid items
        'annotation':'/home/mila/l/le.zhang/scratch/datasets/allava_laion/ALLaVA-Caption-LAION-4V-valid.json',
        'imagedir':'/home/mila/l/le.zhang/scratch/datasets'
        },
    'ALLaVAVFLAN': {
        'annotation':'/home/mila/l/le.zhang/scratch/datasets/allava_vflan/ALLaVA-Caption-VFLAN-4V.json',
        'imagedir':'/home/mila/l/le.zhang/scratch/datasets'
        },
    'coco': {
        'annotation':'/home/mila/l/le.zhang/scratch/datasets/Cambrian-Alignment/jsons/coco.json',
        'imagedir':'/home/mila/l/le.zhang/scratch/datasets/Cambrian-Alignment'
        },
    'sam': {
        'annotation':'/home/mila/l/le.zhang/scratch/datasets/Cambrian-Alignment/jsons/sam.json',
        'imagedir':'/home/mila/l/le.zhang/scratch/datasets/Cambrian-Alignment'
        },
    'Sharegpt4vllava': {
        'annotation':'/home/mila/l/le.zhang/scratch/datasets/Cambrian-Alignment/jsons/llava_pretrain.json',
        'imagedir':'/home/mila/l/le.zhang/scratch/datasets/Cambrian-Alignment'
        },
    'dreamclipcc3m': {
        'annotation':'/home/mila/l/le.zhang/scratch/datasets/DownloadCC3M/cc3m_3long_1raw_captions_filterd.csv',
        'imagedir':'/home/mila/l/le.zhang/scratch/datasets/DownloadCC3M'
        },
    'dreamclipcc12m': {
        'annotation':'/home/mila/l/le.zhang/scratch/datasets/DownloadCC3M/cc12m_3long_3short_1raw_captions_url_path_filtered.csv',
        'imagedir':'/home/mila/l/le.zhang/scratch/datasets/DownloadCC3M/CC12M'
        },
    
}
# argparse for encoding sentences into embeddings and save them
def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True, help='Data type')
    parser.add_argument('--vision_model_name', type=str, required=True, help='Model name')
    parser.add_argument('--text_model_name', type=str, required=True, help='Model name')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--resume', action='store_true', help='Resume from existing embeddings')
    parser.add_argument('--start_index', type=int, default=0, help='Start index for data processing')
    parser.add_argument('--end_index', type=int, default=None, help='End index for data processing')
    parser.add_argument('--domain', type=str, choices=['text', 'image'], required=True, help='Domain to encode')
    return parser.parse_args()



if __name__ == "__main__":
    args = parse_args()

    data_file = DATADIR[args.data]['annotation']
    if data_file.endswith('.json'):
        with open(data_file, 'r') as f:
            data = json.load(f)
        sentences = [sample['conversations'][-1]['value'] for sample in data]
        images = [os.path.join(DATADIR[args.data]['imagedir'], sample['image']) for sample in data]
    elif data_file.endswith('.jsonl'):
        data = []
        with open(data_file, 'r') as f:
            for line in f:
                data.append(json.loads(line))
        sentences = [sample['conversations'][-1]['value'] for sample in data]
        images = [os.path.join(DATADIR[args.data]['imagedir'], sample['image']) for sample in data]
    elif data_file.endswith('.csv'):
        logging.info(f'Loading data from {data_file}...')
        images = []
        sentences = []
        with open(data_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            _ = next(reader)
            for row in reader:
                if args.data == 'dreamclipcc12m':
                    images.append(os.path.join(DATADIR[args.data]['imagedir'], row[-1] + '.jpg'))
                    sentences.append(row[4])
                else:
                    images.append(os.path.join(DATADIR[args.data]['imagedir'], row[4]))
                    sentences.append(row[1])
    else:
        raise ValueError('Unsupported data format')
    
    # 设定开始和结束索引
    start_index = args.start_index
    end_index = args.end_index if args.end_index else len(sentences)
    
    # 仅处理指定范围内的数据
    sentences = sentences[start_index:end_index]
    images = images[start_index:end_index]

    if args.domain == 'text':
        logging.info(f'Encoding text data {args.data} with model {args.text_model_name} of batch size {args.batch_size}...')
        model_name = args.text_model_name.split('/')[-1]
        output_dir = os.path.join('./data/text_embedding', model_name, args.data)
        if not args.resume and os.path.exists(output_dir):
            # logging.info(f'{output_dir} already exists, skipping...')
            exit()
        model = SentenceEmbedding(args.text_model_name)
        model.eval()
        
        # 批处理并编码
        idx = start_index // args.batch_size
        for batch_idx in tqdm(range(0, len(sentences), args.batch_size)):
            output_path = os.path.join(output_dir, f'{idx}.pt')
            if args.resume and os.path.exists(output_path):
                # logging.info(f'{output_path} already exists, skipping...')
                idx += 1
                continue
            batch_sentences = sentences[batch_idx:batch_idx + args.batch_size]
            with torch.no_grad():
                batch_embeddings = model.get_sentence_embeddings(batch_sentences).cpu()
            
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            torch.save(batch_embeddings, output_path)
            torch.cuda.empty_cache()  # 清理缓存
            idx += 1
    else:
        logging.info(f'Encoding image data {args.data} with model {args.vision_model_name} of batch size {args.batch_size}...')
        model_name = args.vision_model_name.split('/')[-1]
        output_dir = os.path.join('./data/image_embedding', model_name, args.data)
        if not args.resume and os.path.exists(output_dir):
            logging.info(f'{output_dir} already exists, skipping...')
            exit()
        model = ImageEmbedding(args.vision_model_name)
        model.eval()
       
        # 批处理并编码
        idx = start_index // args.batch_size
        for batch_idx in tqdm(range(0, len(images), args.batch_size)):
            output_path = os.path.join(output_dir, f'{idx}.pt')
            if args.resume and os.path.exists(output_path):
                logging.info(f'{output_path} already exists, skipping...')
                idx += 1
                continue
            batch_images = images[batch_idx:batch_idx + args.batch_size]
            with torch.no_grad():
                batch_embeddings = model.get_visual_embeddings_from_directory(batch_images).cpu()
            
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            torch.save(batch_embeddings, output_path)
            torch.cuda.empty_cache()  # 清理缓存
            idx += 1
