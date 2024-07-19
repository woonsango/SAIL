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
    
}
# argparse for encoding sentences into embeddings and save them
def parse_args():
    parser = argparse.ArgumentParser(description="Encode sentences into embeddings")
    parser.add_argument("--data", type=str, required=True, choices=DATADIR.keys(), help="data to encode")
    parser.add_argument("--model_name", type=str, default='sentence-transformers/all-mpnet-base-v2', help="model name")
    parser.add_argument("--domain", type=str, choices=['text', 'image'], required=True, help="target type")
    parser.add_argument("--batch_size", type=int, default=5120, help="batch size")    
    parser.add_argument("--resume", action='store_true', help="resume from encoding process")
    return parser.parse_args()



if __name__ == "__main__":
    args = parse_args()

    data_file = DATADIR[args.data]['annotation']
    if data_file.endswith('.json'):
        with open(DATADIR[args.data]['annotation'], 'r') as f:
            data = json.load(f)
    elif data_file.endswith('.jsonl'):
        data = []
        with open(DATADIR[args.data]['annotation'], 'r') as f:
            for line in f:
                data.append(json.loads(line))
    else:
        raise ValueError('Unsupported data format')
    
    if args.domain == 'text':
        logging.info(f'Encoding text data {args.data} with model {args.model_name} of batch size {args.batch_size}...')
        model_name = args.model_name.split('/')[-1]
        output_dir = os.path.join('./data/text_embedding', model_name, args.data)
        if not args.resume and os.path.exists(output_dir):
            logging.info(f'{output_dir} already exists, skipping...')
            exit()
        model = SentenceEmbedding(args.model_name)
        model.eval()
        # load sentences
        sentences = [sample['conversations'][-1]['value'] for sample in data]
        # batchify and encode
        idx=0
        for batch_idx in tqdm(range(0, len(sentences), args.batch_size)):
            output_path = os.path.join(output_dir, f'{idx}.pt')
            if args.resume and os.path.exists(output_path):
                logging.info(f'{output_path} already exists, skipping...')
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
        logging.info(f'Encoding image data {args.data} with model {args.model_name} of batch size {args.batch_size}...')
        model_name = args.model_name.split('/')[-1]
        output_dir = os.path.join('./data/image_embedding', model_name, args.data)
    
        
        if not args.resume and os.path.exists(output_dir):
            logging.info(f'{output_dir} already exists, skipping...')
            exit()
        model = ImageEmbedding(args.model_name)
        model.eval()
        
       
        # load image paths
        images = []
        images = [os.path.join(DATADIR[args.data]['imagedir'], sample['image']) for sample in data]
        # batchify and encode
        idx=0
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


