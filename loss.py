import torch.nn.functional as F
import torch


def clip_loss(vision_embeddings, text_embeddings, temperature=0.07):
    """
    计算 CLIP 损失。

    Args:
    - vision_embeddings: 图像嵌入矩阵 (n, d)
    - text_embeddings: 文本嵌入矩阵 (n, d)
    - temperature: 温度参数，用于调整相似度的平滑度

    Returns:
    - loss: CLIP 损失
    """
    # 归一化嵌入
    vision_embeddings = F.normalize(vision_embeddings, dim=-1)
    text_embeddings = F.normalize(text_embeddings, dim=-1)
    
    # 计算余弦相似度矩阵
    logits_per_image = torch.matmul(vision_embeddings, text_embeddings.t()) / temperature
    logits_per_text = logits_per_image.t()
    
    # 创建标签
    batch_size = vision_embeddings.size(0)
    labels = torch.arange(batch_size, device=vision_embeddings.device)
    
    # 计算交叉熵损失
    loss_image = F.cross_entropy(logits_per_image, labels)
    loss_text = F.cross_entropy(logits_per_text, labels)
    
    # 平均两个损失
    loss = (loss_image + loss_text) / 2.0
    return loss

def sigclip_loss(logits):
    n = logits.size(0)
    labels = 2 * torch.eye(n) - 1  # -1 with diagonal 1
    labels = labels.to(logits.device)
    loss = -torch.mean(F.logsigmoid(labels * logits))
    return loss