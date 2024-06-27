import torch
import os
import json
from tqdm import tqdm
import numpy as np
import json
import argparse
from model import SentenceEmbedding, ImageEmbedding
# argparse for encoding sentences into embeddings and save them
def parse_args():
    parser = argparse.ArgumentParser(description="Encode sentences into embeddings")
    parser.add_argument("--model_name", type=str, default='sentence-transformers/all-mpnet-base-v2', help="model name")
    parser.add_argument("--domain", type=str, choices=['text', 'image'], required=True, help="target type")
    parser.add_argument("--batch_size", type=int, default=2048, help="batch size")    
    return parser.parse_args()




if __name__ == "__main__":
    args = parse_args()
    if args.domain == 'text':
        model = SentenceEmbedding(args.model_name)
        model.eval()
        model_name = model.model.config._name_or_path.split('/')[-1]
        # load sentences
        sentences = []
        with open('./data/blip_laion_cc_sbu_558k.json', 'r') as f:
            data = json.load(f)
        for sample in data:
            sentences.append(sample['conversations'][-1]['value'])
        # batchify and encode
        idx=0
        for batch_idx in tqdm(range(0, len(sentences), args.batch_size)):
            batch_sentences = sentences[batch_idx:batch_idx + args.batch_size]
            with torch.no_grad():
                batch_embeddings = model.get_sentence_embeddings(batch_sentences).cpu()
            output_path = os.path.join('./data/text_embedding', model_name, f'{idx}.pt')
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            torch.save(batch_embeddings, output_path)
            idx += 1
        # save metadata
        # with open(os.path.join('./data/text_embedding', model_name, 'metadata.json'), 'w') as f:
        #     json.dump({'encoded_model_name': model_name, 'chunk_size': args.batch_size, 'total_data_num': len(sentences)}, f)

    else:
        model = ImageEmbedding()
        model.eval()
        model_name = model.model.config._name_or_path.split('/')[-1]
        # load image paths
        images = []
        with open('./data/blip_laion_cc_sbu_558k.json', 'r') as f:
            data = json.load(f)
        for sample in data:
            images.append(os.path.join('./data/image',sample['image']))
        # batchify and encode
        idx=0
        for batch_idx in tqdm(range(0, len(images), args.batch_size)):
            batch_images = images[batch_idx:batch_idx + args.batch_size]
            with torch.no_grad():
                batch_embeddings = model.get_visual_embeddings_from_directory(batch_images).cpu()
        
            output_path = os.path.join('./data/image_embedding', model_name, f'{idx}.pt')
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            torch.save(batch_embeddings, output_path)
            idx += 1
        # save metadata
        # with open(os.path.join('./data/image_embedding', model_name, 'metadata.json'), 'w') as f:
        #     json.dump({'encoded_model_name': model_name, 'chunk_size': args.batch_size, 'total_data_num': len(images)}, f)


