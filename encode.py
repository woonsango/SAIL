import torch
import os
import json
import logging
import csv
from tqdm import tqdm
from model import SentenceEmbedding, ImageEmbedding
from train.logger import setup_logging
from data.data_config import DATADIR
from data.utils import load_data
import warnings
setup_logging(log_file=None, level=logging.INFO)
warnings.filterwarnings("ignore", message="Corrupt EXIF data")
warnings.filterwarnings("ignore", message="Palette images with Transparency")

def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True, help='Data type')
    parser.add_argument('--vision_model_name', type=str, required=True, help='Model name')
    parser.add_argument('--text_model_name', type=str, required=True, help='Model name')
    parser.add_argument('--resume', action='store_true', help='Resume from existing embeddings')
    parser.add_argument('--start_index', type=int, default=0, help='Start index for data processing')
    parser.add_argument('--end_index', type=int, default=None, help='End index for data processing')
    parser.add_argument('--domain', type=str, choices=['text', 'image'], required=True, help='Domain to encode')
    parser.add_argument('--source_caption', type=str, choices=['raw_caption', 'shortIB_captions', 'longIB_captions', 'shortSV_captions', 'longSV_captions', 'shortLLA_captions', 'longLLA_captions','caption'], required=True, help='Source caption')
    parser.add_argument('--batch_size', type=int, default=32, help='Save batch size')
    return parser.parse_args()

def process_batch(data, start_index, batch_size, output_dir, encode_function, resume):
    idx = start_index // batch_size
    for batch_idx in tqdm(range(0, len(data), batch_size)):  # Use batch_size for both encoding and saving
        # Save embeddings in chunks of batch_size with torch.fp16
        output_path = os.path.join(output_dir, f'{idx}.pt')
        if resume and os.path.exists(output_path):
            idx += 1
            continue

        batch_data = data[batch_idx:batch_idx + batch_size]  # Use batch_size for encoding
        with torch.no_grad():
            batch_embeddings = encode_function(batch_data).cpu()  # Encode the entire batch
    
        batch_embeddings = batch_embeddings.half()  # Convert to half precision
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        torch.save(batch_embeddings, output_path)
        idx += 1

def encode_text(args, sentences, start_index):
    model_name = args.text_model_name.split('/')[-1]
    output_dir = os.path.join('./data/tensor_data/text_embedding', model_name, args.data +'_'+ args.source_caption)
    if not args.resume and os.path.exists(output_dir):
        exit()
    model = SentenceEmbedding(args.text_model_name)
    model = model.half().to('cuda')  # Move model to GPU
    model.eval()
    process_batch(sentences, start_index, args.batch_size, output_dir, model.get_sentence_embeddings, args.resume)

def encode_image(args, images, start_index):
    model_name = args.vision_model_name.split('/')[-1]
    output_dir = os.path.join('./data/tensor_data/image_embedding', model_name, args.data)
    if not args.resume and os.path.exists(output_dir):
        logging.info(f'{output_dir} already exists, skipping...')
        exit()
    model = ImageEmbedding(args.vision_model_name)
    model = model.to('cuda')  # Move model to GPU
    model.eval()
    process_batch(images, start_index, args.batch_size, output_dir, model.get_visual_embeddings_from_directory, args.resume)

def main():
    args = parse_args()
    sentences, images = load_data(DATADIR[args.data], args.source_caption, args.domain)
    start_index = args.start_index
    end_index = args.end_index if args.end_index else max(len(sentences), len(images))
    logging.info(f"Start index: {start_index}, End index: {end_index}")
    logging.info(f"Number of sentences: {len(sentences)}")
    logging.info(f"Number of images: {len(images)}")
    # assert begin and end 
    if args.domain == 'text':
        logging.info(f'Encoding text data {args.data} with model {args.text_model_name} of batch size {args.batch_size}...')
        sentences = sentences[start_index:end_index]
        print(f"First 5 items of sentences: {sentences[:5]}")
        encode_text(args, sentences, start_index)
    elif args.domain == 'image':
        logging.info(f'Encoding image data {args.data} with model {args.vision_model_name} of batch size {args.batch_size}...')
        images = images[start_index:end_index]
        print(f"First 5 items of images paths: {images[:5]}")
        encode_image(args, images, start_index)

if __name__ == "__main__":
    main()