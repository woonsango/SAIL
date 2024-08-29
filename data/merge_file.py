import h5py
import numpy as np

def merge_hdf5_files(hdf5_file_list, output_hdf5_file, dataset_name):
    total_samples = 0
    dimension = None

    # 计算总样本数，并获取样本维度
    for hdf5_file in hdf5_file_list:
        with h5py.File(hdf5_file, 'r') as file:
            data = file[dataset_name]
            total_samples += data.shape[0]
            if dimension is None:
                dimension = data.shape[1]

    # 创建合并后的HDF5文件
    with h5py.File(output_hdf5_file, 'w') as output_file:
        merged_dataset = output_file.create_dataset(dataset_name, (total_samples, dimension), dtype='float32')

        current_index = 0
        # 将每个文件的数据写入合并后的文件
        for hdf5_file in hdf5_file_list:
            with h5py.File(hdf5_file, 'r') as file:
                data = file[dataset_name]
                num_samples = data.shape[0]
                merged_dataset[current_index:current_index + num_samples] = data[:]
                current_index += num_samples

    print(f"Successfully merged files into {output_hdf5_file}")

# 合并text和image的HDF5文件
text_hdf5_file_list = ['/home/mila/l/le.zhang/scratch/light_align/data/text_embedding/gte-large-en-v1.5/dreamclipcc3m_longSV_captions.h5', '/home/mila/l/le.zhang/scratch/light_align/data/text_embedding/gte-large-en-v1.5/dreamclipcc12m_longSV_captions.h5']
image_hdf5_file_list = ['/home/mila/l/le.zhang/scratch/light_align/data/image_embedding/dinov2-large/dreamclipcc3m_images.h5', '/home/mila/l/le.zhang/scratch/light_align/data/image_embedding/dinov2-large/dreamclipcc12m_images.h5']

# merge_hdf5_files(text_hdf5_file_list, '/home/mila/l/le.zhang/scratch/light_align/data/text_embedding/gte-large-en-v1.5/dreamclipcc15m_longSV_captions.h5', 'data')
merge_hdf5_files(image_hdf5_file_list, '/home/mila/l/le.zhang/scratch/light_align/data/image_embedding/dinov2-large/dreamclipcc15m_images.h5', 'data')
