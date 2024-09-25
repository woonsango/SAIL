import torch
import random

import pandas as pd


import torch
from transformers import AutoImageProcessor, AutoModel
from PIL import Image
import requests
import os
import random
import torch.nn.functional as F
import time

test = 'dimension'
if test == "embedding":
    csv = pd.read_csv('/home/mila/l/le.zhang/scratch/datasets/LAION/30M_laion_synthetic_filtered_large_with_path_filtered.csv', nrows = 10000000)
    csv.head()
    processor = AutoImageProcessor.from_pretrained('facebook/dinov2-large')
    model = AutoModel.from_pretrained('facebook/dinov2-large',attn_implementation="sdpa")
    # model = AutoModel.from_pretrained('facebook/dinov2-large')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.half().to(device)
    # model = torch.compile(model)
    model.eval()

    def compare_image_embeddings(model,processor,n_tests=5, verbose=False):
        differences = []
        all_test_indices = []
        device = model.device

        for _ in range(n_tests):
            file_idx = random.randint(0, 2000)
            idx = random.randint(0, 4095)  # Changed to 0-4095 to match the tensor size
            test_index = file_idx * 4096 + idx
            all_test_indices.append(test_index)

            image_path = os.path.join("/home/mila/l/le.zhang/scratch/datasets/LAION", csv.iloc[test_index]['Image Path'])
            image = Image.open(image_path)
            
            if verbose:
                print(f"Test {_ + 1}:")
                print(f"Comparing image at index {test_index}")

            # Load pre-computed embedding
            x = torch.load(f"/home/mila/l/le.zhang/scratch/light_align/data/tensor_data/image_embedding/dinov2-large/laion30m/{file_idx}.pt", weights_only=True)
            precomputed_embedding = x[idx].to(device)

            # Compute embedding on-the-fly
            inputs = processor(images=image, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model(**inputs)
            sequence_output = outputs[0]
            cls_token = sequence_output[:, 0]
            patch_tokens = sequence_output[:, 1:]
            computed_embedding = torch.cat([cls_token, patch_tokens.mean(dim=1)], dim=1).to(torch.float16)[0]

            # Move embeddings back to CPU for comparison and storage
            precomputed_embedding = precomputed_embedding.cpu()
            computed_embedding = computed_embedding.cpu()

            if verbose:
                print(f"Precomputed embedding: {precomputed_embedding}")
                print(f"Computed embedding: {computed_embedding}")

            # Compute the difference
            diff = F.mse_loss(precomputed_embedding, computed_embedding).item()
            differences.append(diff)
            
            if verbose:
                print(f"Difference (MSE): {diff}\n")

        # Print summary statistics
        print(f"All test indices: {all_test_indices}")
        print(f"Average difference over {n_tests} tests: {sum(differences) / n_tests}")
        print(f"Max difference: {max(differences)}")
        print(f"Min difference: {min(differences)}")

    start_time = time.time()
    # Usage example:
    compare_image_embeddings(model,processor,n_tests=1000, verbose=False)
    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")

else:
    import os
    import glob
    import torch
    from tqdm import tqdm
    file_dir = "/home/mila/l/le.zhang/scratch/light_align/data/tensor_data/image_embedding/dinov2-large/laion30m"
    files = glob.glob(os.path.join(file_dir, "*.pt"))
    total_len = 0
    for file in files:
        try:
            tensor = torch.load(file, map_location=torch.device('cpu'), weights_only=True)
            total_len += tensor.size(0)
            if tensor.size(0) != 4096:
                print(file, tensor.size(0))
        except Exception as e:
            print(file, e)
    print(total_len)

