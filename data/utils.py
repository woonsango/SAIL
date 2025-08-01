import os
import json
import csv
import torch
from tqdm import tqdm
import itertools    
import time
import pandas as pd


# instruction = "Summarize the following sentences: "
instruction = ""

def load_data(data_config, source_caption, domain):
    data_file = data_config['annotation']
    if data_file.endswith('.json'):
        if data_config['imagedir'].endswith("train2017"):
            return load_json_data_coco(data_file, data_config['imagedir'])
        else:
            return load_json_data(data_file, data_config['imagedir'])
    elif data_file.endswith('.jsonl'):
        return load_jsonl_data(data_file, data_config['imagedir'])
    elif data_file.endswith('.csv'):
        return load_csv_data(data_file, data_config['imagedir'], source_caption, domain)
    else:
        raise ValueError('Unsupported data format')
    
def load_json_data_coco(data_file, image_dir):
    with open(data_file, 'r') as f:
        data = json.load(f)
    
    id_to_filename = {img["id"]: img["file_name"] for img in data["images"]}

    # 결과 리스트 생성
    sentences = []
    images = []

    # annotations를 순회하면서 매칭되는 file_name과 caption을 추출
    for anno in data["annotations"]:
        image_id = anno["image_id"]
        caption = anno["caption"]
        
        # image_id에 해당하는 file_name이 존재할 경우만 처리
        if image_id in id_to_filename:
            file_name = id_to_filename[image_id]
            file_name = os.path.join(image_dir,file_name)
            sentences.append(caption)
            images.append(file_name)
    
    return sentences, images
    

def load_json_data(data_file, image_dir):
    with open(data_file, 'r') as f:
        data = json.load(f)
    sentences = [sample['conversations'][-1]['value'] for sample in data]
    images = [os.path.join(image_dir, sample['image']) for sample in data]
    return sentences, images

def load_jsonl_data(data_file, image_dir):
    data = []
    with open(data_file, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    sentences = [sample['conversations'][-1]['value'] for sample in data]
    images = [os.path.join(image_dir, sample['image']) for sample in data]
    return sentences, images

def read_csv_column(file_path, column_name, chunksize=None):
    start_time = time.time()  # Start timing
    column_data = []
    
    # Choose reading method based on whether chunksize is used
    if chunksize:
        # Read in chunks
        for chunk in pd.read_csv(file_path, usecols=[column_name], chunksize=chunksize):
            column_data.extend(chunk[column_name].tolist())
    else:
        # Read directly
        column_data = pd.read_csv(file_path, usecols=[column_name])[column_name].tolist()
    
    end_time = time.time()  # End timing
    elapsed_time = end_time - start_time
    
    # Output results and runtime
    print(f"Time taken to read '{column_name}' column: {elapsed_time:.2f} seconds")
    return column_data

def load_csv_data(data_file, image_dir, source_caption, domain):
    # Read image paths and captions using read_csv_column function
    images = []
    sentences = []
    if domain == 'image':
        image_paths = read_csv_column(data_file, 'Image Path')
        for image_path in image_paths:
            try:
                full_image_path = os.path.join(image_dir, image_path)
                images.append(full_image_path)
            except Exception as e:
                print(f"Error processing image path: {e}")
                images.append(None)  # Append None for failed image paths
    elif domain == 'text':
        sentences = read_csv_column(data_file, source_caption)
        sentences = [instruction + sentence for sentence in sentences]
    
    return sentences, images
