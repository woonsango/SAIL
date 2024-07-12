import torch
import os
import json
from tqdm import tqdm
import numpy as np
import json
import argparse
from model import SentenceEmbedding, ImageEmbedding
import logging

logging.basicConfig(level=logging.INFO)
# set up logging format
logging.basicConfig(format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')


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
}
# argparse for encoding sentences into embeddings and save them
def parse_args():
    parser = argparse.ArgumentParser(description="Encode sentences into embeddings")
    parser.add_argument("--data", type=str, required=True, choices=DATADIR.keys(), help="data to encode")
    parser.add_argument("--model_name", type=str, default='sentence-transformers/all-mpnet-base-v2', help="model name")
    parser.add_argument("--domain", type=str, choices=['text', 'image'], required=True, help="target type")
    parser.add_argument("--batch_size", type=int, default=5120, help="batch size")    
    return parser.parse_args()



if __name__ == "__main__":
    args = parse_args()

    with open(DATADIR[args.data]['annotation'], 'r') as f:
        data = json.load(f)
    if args.domain == 'text':
        logging.info(f'Encoding text data {args.data} with model {args.model_name} of batch size {args.batch_size}...')
        output_dir = os.path.join('./data/text_embedding', args.model_name.split('/')[-1], args.data)
        if os.path.exists(output_dir):
            logging.info(f'{output_dir} already exists, skipping...')
            exit()
        model = SentenceEmbedding(args.model_name)
        model.eval()
        model_name = model.model.config._name_or_path.split('/')[-1]
        # load sentences
        sentences = []
        
        for sample in data:
            sentences.append(sample['conversations'][-1]['value'])
        # batchify and encode
        idx=0
        for batch_idx in tqdm(range(0, len(sentences), args.batch_size)):
            batch_sentences = sentences[batch_idx:batch_idx + args.batch_size]
            with torch.no_grad():
                batch_embeddings = model.get_sentence_embeddings(batch_sentences).cpu()
            output_path = os.path.join(output_dir, f'{idx}.pt')
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            torch.save(batch_embeddings, output_path)
            idx += 1
    else:
        logging.info(f'Encoding image data {args.data} with model {args.model_name} of batch size {args.batch_size}...')
        model_name = args.model_name.split('/')[-1]
        output_dir = os.path.join('./data/image_embedding', model_name, args.data)
        if os.path.exists(output_dir):
            logging.info(f'{output_dir} already exists, skipping...')
            exit()
        model = ImageEmbedding(args.model_name)
        model.eval()
        
       
        # load image paths
        images = []
        for sample in data:
            images.append(os.path.join(DATADIR[args.data]['imagedir'],sample['image']))
        # batchify and encode
        idx=0
        for batch_idx in tqdm(range(0, len(images), args.batch_size)):
            batch_images = images[batch_idx:batch_idx + args.batch_size]
            with torch.no_grad():
                batch_embeddings = model.get_visual_embeddings_from_directory(batch_images).cpu()
        
            output_path = os.path.join(output_dir, f'{idx}.pt')
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            torch.save(batch_embeddings, output_path)
            idx += 1
        # save metadata
        # with open(os.path.join('./data/image_embedding', model_name, 'metadata.json'), 'w') as f:
        #     json.dump({'encoded_model_name': model_name, 'chunk_size': args.batch_size, 'total_data_num': len(images)}, f)


