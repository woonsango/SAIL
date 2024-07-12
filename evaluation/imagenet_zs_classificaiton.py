import sys
sys.path.append("/home/mila/l/le.zhang/scratch/light_align")
from .imagenetv2 import ImageNetV2Dataset
from .imagenet_constant import IMAGENET_CLASSES, IMAGENET_TEMPLATES
import torch
from model import VLContrastModel
from tqdm import tqdm
import os
import json
import clip
from typing import Union, Optional
import torch.nn as nn
from .utils import get_model_device

os.environ["TOKENIZERS_PARALLELISM"] = "false"

class Processor:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, images):
        images = self.processor(images, return_tensors="pt")["pixel_values"][0]
        return images


def accuracy(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [
        float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy().item())
        for k in topk
    ]

def zeroshot_classifier(model, device, classnames, templates, tokenizer):
    
    with torch.no_grad():
        zeroshot_weights = []
        for classname in tqdm(classnames):
            texts = [
                template.format(classname) for template in templates
            ]  # format with class
            texts = tokenizer(
                texts, padding=True, truncation=True, return_tensors="pt"
            ).to(
                device
            )  # tokenize
            class_embeddings = model.encode_text(texts)  # embed with text encoder
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).cuda()
    return zeroshot_weights

def imagenet_eval(
        model: nn.Module,
        bs: int = 1024, 
        images_dir: str = "/home/mila/l/le.zhang/scratch/datasets"
):
    model.eval()
    device = get_model_device(model)
    tokenizer = model.text_model.tokenizer
    processor = Processor(model.vision_model.image_processor)
    zeroshot_weights = zeroshot_classifier(model, device, IMAGENET_CLASSES, IMAGENET_TEMPLATES, tokenizer)
    images = ImageNetV2Dataset(
        variant="matched-frequency",
        transform=processor,
        location=images_dir,
    )
    loader = torch.utils.data.DataLoader(images, batch_size=bs, num_workers=2)
    with torch.no_grad():
        top1, top5, n = 0.0, 0.0, 0.0
        for i, (images, target) in enumerate(tqdm(loader)):
            images = images.to(device)
            target = target.to(device)

            # predict
            image_features = model.encode_image({"pixel_values": images})
            image_features /= image_features.norm(dim=-1, keepdim=True)
            logits = 100.0 * image_features @ zeroshot_weights

            # measure accuracy
            acc1, acc5 = accuracy(logits, target, topk=(1, 5))
            top1 += acc1
            top5 += acc5
            n += images.size(0)

    top1 = (top1 / n) * 100
    top5 = (top5 / n) * 100
    print(f"Top-1 accuracy: {top1:.2f}")
    print(f"Top-5 accuracy: {top5:.2f}")

    return {"top1": top1, "top5": top5}

if __name__ == "__main__":
    # Load the model
    linear_align = True
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    if linear_align:
        model = VLContrastModel(
            text_model_name="sentence-transformers/all-mpnet-base-v2",
            vision_model_name="facebook/dinov2-base",
            vlhead_weights_path = '/home/mila/l/le.zhang/scratch/light_align/logs/single_node_test_20240710124601_bs_16384_lion_mean_0.1_warmup/checkpoints/epoch_100.pt',
            linear_align=True,
        )
        # weights_path='/home/mila/l/le.zhang/scratch/light_align/output/raw_data_linear/model_14.pth'
    else:
        model = VLContrastModel(
            text_model_name="sentence-transformers/all-mpnet-base-v2",
            vision_model_name="facebook/dinov2-base",
            vlhead_weights_path = '/home/mila/l/le.zhang/scratch/light_align/output/raw_data_linear_shared_head/checkpoint_42.pth',
            linear_align=False,
        )
    model = model.to(device)
    imagenet_eval(model)