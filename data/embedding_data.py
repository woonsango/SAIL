import torch
from torch.utils.data import Dataset, DataLoader
import os
from tqdm import tqdm
import glob
from natsort import natsorted
import os
import torch
from torch.utils.data import Dataset, DataLoader
import json
from threading import Lock
from torch.nn.utils.rnn import pad_sequence

def custom_collate_fn(batch):
    text_vectors, image_vectors = zip(*batch)
    
    text_vectors_padded = pad_sequence(text_vectors, batch_first=True)
    image_vectors_padded = pad_sequence(image_vectors, batch_first=True)
    
    return text_vectors_padded, image_vectors_padded

class VLEmbeddingDataset(Dataset):
    def __init__(self, text_embedding_list, image_embedding_list):
        self.text_embedding_dir = text_embedding_list
        self.image_embedding_dir = image_embedding_list
        
        # Note: must sort the file names to ensure the correspondence of text and image vectors
        self.text_files = []
        for dir_path in text_embedding_list:
            files = glob.glob(os.path.join(dir_path, "*.pt"))
            sorted_files = natsorted(files)
            self.text_files.extend(sorted_files)
        
        self.image_files = []
        for dir_path in image_embedding_list:
            files = glob.glob(os.path.join(dir_path, "*.pt"))
            sorted_files = natsorted(files)
            self.image_files.extend(sorted_files)
        
        
        self.text_vectors = [vector for file in self.text_files for vector in torch.load(file)]
        self.image_vectors = [vector for file in self.image_files for vector in torch.load(file)]
        
        assert len(self.text_vectors) == len(self.image_vectors), f"text and image vectors have different lengths, {len(self.text_vectors)} vs {len(self.image_vectors)}"
        
        self.total_length = len(self.text_vectors)
        self.visual_dim = self.image_vectors[0].shape[0]
        self.text_dim = self.text_vectors[0].shape[0]
    
    def __len__(self):
        return self.total_length
    
    def __getitem__(self, idx):
        return self.text_vectors[idx], self.image_vectors[idx]


if __name__ == "__main__":

    text_embedding_dir = '/home/mila/l/le.zhang/scratch/light_align/data/text_embedding/all-mpnet-base-v2'
    image_embedding_dir = '/home/mila/l/le.zhang/scratch/light_align/data/image_embedding/dinov2-base'
    print("Loading dataset...")
    # dataset = LazyVLEmbeddingDataset(text_embedding_dir, image_embedding_dir)
    dataset = VLEmbeddingDataset(text_embedding_dir, image_embedding_dir)
    print("Dataset loaded.")
    # 创建DataLoader
    breakpoint()
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=4)
    for text_vectors, image_vectors in tqdm(dataloader):
        print(text_vectors.shape, image_vectors.shape)
        
