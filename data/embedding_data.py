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
import h5py
import numpy as np

def custom_collate_fn(batch):
    text_vectors, image_vectors = zip(*batch)

    # if len(text_vectors[0].shape) == 1:
    #     text_vectors = torch.stack(text_vectors, dim=0)
    #     image_vectors = torch.stack(image_vectors, dim=0)
    # elif len(text_vectors[0].shape) == 2:
    #     text_vectors = torch.cat(text_vectors, dim=0)
    #     image_vectors = torch.cat(image_vectors, dim=0)
    # else:
    #     raise ValueError("Unsupported shape of text vectors")

    text_vectors = pad_sequence(text_vectors, batch_first=True, padding_value=0)
    image_vectors = pad_sequence(image_vectors, batch_first=True, padding_value=0)

    
    
    return text_vectors, image_vectors

class VLEmbeddingDataset(Dataset):
    def __init__(self, text_embedding_list, image_embedding_list, train_num_samples=None):
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
        
        
        self.text_vectors = [vector for file in self.text_files for vector in torch.load(file, weights_only=True).to(torch.float16)]
        self.image_vectors = [vector for file in self.image_files for vector in torch.load(file, weights_only=True).to(torch.float16)]
        
        assert len(self.text_vectors) % len(self.image_vectors) == 0, f"text vectors length ({len(self.text_vectors)}) is not a multiple of image vectors length ({len(self.image_vectors)})"

        if train_num_samples is not None:
            num_samples = len(self.text_vectors)
            random_indices = np.random.choice(num_samples, train_num_samples, replace=False)
            self.text_vectors = [self.text_vectors[i] for i in random_indices]
            self.image_vectors = [self.image_vectors[i] for i in random_indices]
            print(f"Random Selecting {train_num_samples} samples as training data")

        self.image_num = len(self.image_vectors)
        self.text_num = len(self.text_vectors)

        self.visual_dim = self.image_vectors[0].shape[-1]
        self.text_dim = self.text_vectors[0].shape[-1]
    
    def __len__(self):
        return self.text_num
    
    def __getitem__(self, idx):
        # multiple text for one image
        if idx >= self.image_num:
            img_idx = idx % self.image_num
        else:
            img_idx = idx
        return self.text_vectors[idx], self.image_vectors[img_idx]


# class VLEmbeddingDataset(Dataset):
#     def __init__(self, text_embedding_list, image_embedding_list):
#         self.text_embedding_dir = text_embedding_list
#         self.image_embedding_dir = image_embedding_list
        
#         # 建立文本和图像的文件列表，确保文件名排序对应
#         self.text_files = []
#         for dir_path in text_embedding_list:
#             files = glob.glob(os.path.join(dir_path, "*.pt"))
#             sorted_files = natsorted(files)
#             self.text_files.extend(sorted_files)
        
#         self.image_files = []
#         for dir_path in image_embedding_list:
#             files = glob.glob(os.path.join(dir_path, "*.pt"))
#             sorted_files = natsorted(files)
#             self.image_files.extend(sorted_files)
        
#         assert len(self.text_files) == len(self.image_files), f"text and image files have different lengths, {len(self.text_files)} vs {len(self.image_files)}"
        
#         self.total_length = len(self.text_files)
#         self.visual_dim = torch.load(self.image_files[0], weights_only=True).shape[-1]
#         self.text_dim = torch.load(self.text_files[0], weights_only=True).shape[-1]
#         print(f"Visual dim: {self.visual_dim}, Text dim: {self.text_dim}")
#         print(f"Total samples: {self.total_length}")
#         print(self.text_files[0])
    
#     def __len__(self):
#         return self.total_length
    
#     def __getitem__(self, idx):
#         # 动态加载数据
#         text_vector = torch.load(self.text_files[idx], weights_only=True)
#         image_vector = torch.load(self.image_files[idx], weights_only=True)
        
#         return text_vector, image_vector


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
        
