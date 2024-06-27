import torch
from torch.utils.data import Dataset, DataLoader
import os
from tqdm import tqdm


import os
import torch
from torch.utils.data import Dataset, DataLoader
import json
from threading import Lock



class LazyVLEmbeddingDataset(Dataset):
    def __init__(self, text_embedding_dir, image_embedding_dir, max_cache_size=5):
        self.text_embedding_dir = text_embedding_dir
        self.image_embedding_dir = image_embedding_dir
        self.max_cache_size = max_cache_size
        self.text_files = []
        self.image_files = []
        self.text_lengths = []
        self.image_lengths = []
        self.text_cache = {}
        self.image_cache = {}
        self.cache_lock = Lock()
        
        # 遍历目录中的文件并分别加载文本和图像向量文件
        for file in os.listdir(text_embedding_dir):
            self.text_files.append(os.path.join(text_embedding_dir, file))
        
        for file in os.listdir(image_embedding_dir):
            self.image_files.append(os.path.join(image_embedding_dir, file))
        
        # 确保文件数量一致
        assert len(self.text_files) == len(self.image_files), "文本和图像向量文件数量不一致"
        
        # 计算每个文件的长度，并存储在列表中
        for file in self.text_files:
            text_vectors = torch.load(file)
            self.text_lengths.append(len(text_vectors))
            
        for file in self.image_files:
            image_vectors = torch.load(file)
            self.image_lengths.append(len(image_vectors))
        
        assert sum(self.text_lengths) == sum(self.image_lengths), "文本和图像向量的总数不一致"
        
        self.total_length = sum(self.text_lengths)
    
    def __len__(self):
        return self.total_length
    
    def __getitem__(self, idx):
        # 确定索引属于哪个文件
        file_idx, vector_idx = self._get_file_and_vector_index(idx)
        
        text_file = self.text_files[file_idx]
        image_file = self.image_files[file_idx]
        
        with self.cache_lock:
            if text_file not in self.text_cache:
                if len(self.text_cache) >= self.max_cache_size:
                    self._clear_cache(self.text_cache)
                self.text_cache[text_file] = torch.load(text_file)
            
            if image_file not in self.image_cache:
                if len(self.image_cache) >= self.max_cache_size:
                    self._clear_cache(self.image_cache)
                self.image_cache[image_file] = torch.load(image_file)
        
        text_vector = self.text_cache[text_file][vector_idx]
        image_vector = self.image_cache[image_file][vector_idx]
        
        return text_vector, image_vector
    
    def _get_file_and_vector_index(self, idx):
        current_length = 0
        for i, length in enumerate(self.text_lengths):
            if idx < current_length + length:
                return i, idx - current_length
            current_length += length
        raise IndexError("索引超出范围")
    
    def _clear_cache(self, cache):
        # 清理缓存中的最早的一个文件
        first_key = next(iter(cache))
        del cache[first_key]

if __name__ == "__main__":

    text_embedding_dir = '/home/mila/l/le.zhang/scratch/light_align/data/text_embedding/all-mpnet-base-v2'
    image_embedding_dir = '/home/mila/l/le.zhang/scratch/light_align/data/image_embedding/dinov2-base'
    print("Loading dataset...")
    dataset = LazyVLEmbeddingDataset(text_embedding_dir, image_embedding_dir)
    print("Dataset loaded.")
    # 创建DataLoader
    breakpoint()
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=4)
    for text_vectors, image_vectors in tqdm(dataloader):
        print(text_vectors.shape, image_vectors.shape)
        
