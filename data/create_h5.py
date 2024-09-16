import os
import torch
import h5py
from tqdm import tqdm
from glob import glob

def convert_pt_to_h5(pt_file_list, output_h5_path):

    with h5py.File(output_h5_path, 'w') as hdf:
        tensor_counter = 0  # 用于给每个张量命名
        for pt_file in tqdm(pt_file_list, desc=f"Processing {os.path.basename(output_h5_path)}"):
            tensors = torch.load(pt_file, weights_only=True)
            for tensor in tensors:
                hdf.create_dataset(f'tensor_{tensor_counter}', data=tensor.numpy())
                tensor_counter += 1
    print(f"All .pt files have been successfully converted to HDF5 file!")

if __name__ == "__main__":
    folder_list = ["/home/mila/l/le.zhang/scratch/light_align/data/tensor_data/image_embedding/dinov2-large/dreamclipcc3m", "/home/mila/l/le.zhang/scratch/light_align/data/tensor_data/image_embedding/dinov2-large/dreamclipcc12mhf", "/home/mila/l/le.zhang/scratch/light_align/data/tensor_data/text_embedding/gte-large-en-v1.5/dreamclipcc3m_raw", "/home/mila/l/le.zhang/scratch/light_align/data/tensor_data/text_embedding/gte-large-en-v1.5/dreamclipcc3m_longSV", "/home/mila/l/le.zhang/scratch/light_align/data/tensor_data/text_embedding/gte-large-en-v1.5/dreamclipcc12mhf_raw_caption", "/home/mila/l/le.zhang/scratch/light_align/data/tensor_data/text_embedding/gte-large-en-v1.5/dreamclipcc12mhf_longSV_captions"]

    for pt_folder in folder_list:
        print(f"Processing {pt_folder}")
        pt_file_list = glob(os.path.join(pt_folder, '*.pt'))
        if 'image' in pt_folder:
            output_h5_dir = "/home/mila/l/le.zhang/scratch/light_align/data/hdf5_data/image_h5"
        else:
            output_h5_dir = "/home/mila/l/le.zhang/scratch/light_align/data/hdf5_data/text_h5"
        model_name = pt_folder.split('/')[-2]
        dataset_name = pt_folder.split('/')[-1]
        print(f"Processing {model_name} {dataset_name}")
        convert_pt_to_h5(pt_file_list, os.path.join(output_h5_dir, f"{model_name}_{dataset_name}.h5"))