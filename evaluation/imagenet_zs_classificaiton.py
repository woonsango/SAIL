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


def zeroshot_classifier(model, device, classnames, templates, tokenizer, model_name):

    with torch.no_grad():
        zeroshot_weights = []
        # check if ./evaluation/backbone_features/{model_name}/{evaluation_data}.pt exists
        if os.path.exists(f"./evaluation/backbone_features/{model_name}/imagenet.pt"):
            print("Loading backbone features from disk")
            pre_encode_model_featuers = torch.load(
                f"./evaluation/backbone_features/{model_name}/imagenet.pt"
            )
            for classname in tqdm(classnames):
                pre_encode_model_featuers_class = pre_encode_model_featuers[
                    classname
                ].to(device)
                class_embeddings = model.encode_text_head(
                    pre_encode_model_featuers_class
                )  # embed with head only
                class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
                class_embedding = class_embeddings.mean(dim=0)
                class_embedding /= class_embedding.norm()
                zeroshot_weights.append(class_embedding)
        else:
            print("Extracting backbone features")
            pre_encode_model_featuers = {}
            for classname in tqdm(classnames):
                texts = [
                    template.format(classname) for template in templates
                ]  # format with class
                texts = tokenizer(
                    texts, padding=True, truncation=True, return_tensors="pt"
                ).to(
                    device
                )  # tokenize
                class_embeddings, pre_encode_model_featuer = (
                    model.encode_text_return_encodedFeatures(texts)
                )  # embed with head only
                pre_encode_model_featuers[classname] = pre_encode_model_featuer.cpu()
                class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
                class_embedding = class_embeddings.mean(dim=0)
                class_embedding /= class_embedding.norm()
                zeroshot_weights.append(class_embedding)
            os.makedirs(f"./evaluation/backbone_features/{model_name}", exist_ok=True)
            torch.save(
                pre_encode_model_featuers,
                f"./evaluation/backbone_features/{model_name}/imagenet.pt",
            )
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).cuda()
    return zeroshot_weights


def imagenet_eval(
    model: nn.Module,
    bs: int = 1024,
    images_dir: str = "/home/mila/l/le.zhang/scratch/datasets",
    text_model_name: str = "sentence-transformers/all-mpnet-base-v2",
    vision_model_name: str = "facebook/dinov2-base",
):
    model.eval()
    device = get_model_device(model)
    tokenizer = model.text_model.tokenizer
    processor = Processor(model.vision_model.image_processor)
    zeroshot_weights = zeroshot_classifier(
        model, device, IMAGENET_CLASSES, IMAGENET_TEMPLATES, tokenizer, text_model_name
    )
    top1, top5, n = 0.0, 0.0, 0.0
    if not os.path.exists(
        f"./evaluation/backbone_features/{vision_model_name}/imagenet.pt"
    ):
        print("Extracting backbone features")
        images_dataset = ImageNetV2Dataset(
            variant="matched-frequency",
            transform=processor,
            location=images_dir,
        )
        loader = torch.utils.data.DataLoader(
            images_dataset, batch_size=bs, num_workers=2
        )
        pre_encode_image_features = {}
        with torch.no_grad():
            for i, (images, target, image_name) in enumerate(tqdm(loader)):
                images = images.to(device)
                target = target.to(device)

                # predict
                image_features, encoded_features = (
                    model.encode_image_return_encodedFeatures({"pixel_values": images})
                )
                for j, name in enumerate(image_name):
                    pre_encode_image_features[name] = {
                        "features": encoded_features[j].cpu(),
                        "target": target[j].cpu(),
                    }
                image_features /= image_features.norm(dim=-1, keepdim=True)
                logits = 100.0 * image_features @ zeroshot_weights

                # measure accuracy
                acc1, acc5 = accuracy(logits, target, topk=(1, 5))
                top1 += acc1
                top5 += acc5
                n += images.size(0)
        os.makedirs(
            f"./evaluation/backbone_features/{vision_model_name}", exist_ok=True
        )
        with open(
            f"./evaluation/backbone_features/{vision_model_name}/imagenet.pt", "wb"
        ) as f:
            torch.save(pre_encode_image_features, f)
    else:
        print("Loading backbone features from disk")
        pre_encode_image_features = torch.load(
            f"./evaluation/backbone_features/{vision_model_name}/imagenet.pt"
        )
        # split the pre_encode_image_features into batches
        batched_pre_encode_image_features = {}
        for i, (key, value) in enumerate(pre_encode_image_features.items()):
            if i % bs == 0:
                batched_pre_encode_image_features[i // bs] = {}
            batched_pre_encode_image_features[i // bs][key] = value
        for i, batch in tqdm(batched_pre_encode_image_features.items()):
            encoded_features = []
            targets = []
            for key, value in batch.items():
                encoded_features.append(value["features"])
                targets.append(value["target"])
            encoded_features = torch.stack(encoded_features).to(device)
            targets = torch.stack(targets).to(device)
            image_features = model.encode_image_head(encoded_features)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            logits = 100.0 * image_features @ zeroshot_weights
            acc1, acc5 = accuracy(logits, targets, topk=(1, 5))
            top1 += acc1
            top5 += acc5
            n += encoded_features.size(0)

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
            vlhead_weights_path="/home/mila/l/le.zhang/scratch/light_align/logs/single_node_test_20240710124601_bs_16384_lion_mean_0.1_warmup/checkpoints/epoch_100.pt",
            linear_align=True,
        )
        # weights_path='/home/mila/l/le.zhang/scratch/light_align/output/raw_data_linear/model_14.pth'
    else:
        model = VLContrastModel(
            text_model_name="sentence-transformers/all-mpnet-base-v2",
            vision_model_name="facebook/dinov2-base",
            vlhead_weights_path="/home/mila/l/le.zhang/scratch/light_align/output/raw_data_linear_shared_head/checkpoint_42.pth",
            linear_align=False,
        )
    model = model.to(device)
    imagenet_eval(model)
