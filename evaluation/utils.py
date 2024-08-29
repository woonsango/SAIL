import json
import re
import os
import torch

def check_epoch_exists(json_file, epoch):
    """
    检查 JSON 文件中是否已存在指定 epoch 的结果。
    
    :param json_file: JSON 文件路径
    :param epoch: 要检查的 epoch 编号
    :return: 如果已存在则返回 True，否则返回 False
    """
    if not os.path.exists(json_file):
        return False
    
    try:
        with open(json_file, 'r') as file:
            data = json.load(file)
    except json.JSONDecodeError:
        # 如果文件存在但不是有效的 JSON 格式
        return False

    return f'epoch_{epoch}' in data

def update_results_json(json_file, epoch, results):
    try:
        with open(json_file, 'r') as file:
            data = json.load(file)
    except FileNotFoundError:
        data = {}

    data[f'epoch_{epoch}'] = results
    os.makedirs(os.path.dirname(json_file), exist_ok=True)
    with open(json_file, 'w') as file:
        json.dump(data, file, indent=4)

def extract_info_from_path(path):
    # 提取 epoch 后的数字
    epoch_pattern = r'epoch_(\d+)\.pt'
    epoch_match = re.search(epoch_pattern, path)
    epoch_number = epoch_match.group(1) if epoch_match else None

    # 提取从 bs 开始到 / 结束的字符串
    bs_pattern = r'bs_[^/]+'
    bs_match = re.search(bs_pattern, path)
    bs_string = bs_match.group(0) if bs_match else None

    # 提取在 _bs 之前的字符串
    prefix_pattern = r'logs/([^/]+)_bs_'
    prefix_match = re.search(prefix_pattern, path)
    prefix_string = prefix_match.group(1) if prefix_match else None

    return epoch_number, bs_string, prefix_string

def get_model_device(model):
    return next(model.parameters()).device


# Encodes all text and images in a dataset
def save_features(features, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "wb") as f:
        torch.save(features, f)

def load_features(save_path):
    with open(save_path, "rb") as f:
        return torch.load(f)
    
def grouped_mean_pooling(tensor, m):
    """
    在d维度上将张量分成m组，并对每组进行平均池化。

    参数:
    tensor (torch.Tensor): 输入张量，形状为 (n, d)。
    m (int): 分组的数量，d 必须能被 m 整除。

    返回:
    torch.Tensor: 平均池化后的张量，形状为 (n, m)。
    """
    n, d = tensor.shape

    # 检查 d 是否能被 m 整除
    assert d % m == 0, "d 维度必须能被 m 整除"

    # 计算每组的大小
    group_size = d // m

    # 重塑张量，以便在 m 组上进行平均池化
    tensor_reshaped = tensor.view(n, m, group_size)

    # 对每组进行平均池化
    pooled_result = tensor_reshaped.mean(dim=-1)

    return pooled_result