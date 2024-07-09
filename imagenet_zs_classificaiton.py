from imagenetv2 import ImageNetV2Dataset
from data import IMAGENET_CLASSES, IMAGENET_TEMPLATES
import torch
from model import VLContrastModel
from tqdm import tqdm
import os
import json


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
        float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy())
        for k in topk
    ]


def zeroshot_classifier(classnames, templates):
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


def get_zsshot_weights(model, classnames, templates):
    with torch.no_grad():
        zeroshot_weights = []
        for classname in tqdm(classnames):
            texts = [
                template.format(classname) for template in templates
            ]  # format with class


if __name__ == "__main__":
    # Load the model
    linear = False
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    if linear:
        model = VLContrastModel(
            text_model_name="sentence-transformers/all-mpnet-base-v2",
            vision_model_name="facebook/dinov2-base",
            device=device,
            linear=True,
        )
        # weights_path='/home/mila/l/le.zhang/scratch/light_align/output/raw_data_linear/model_14.pth'
    else:
        model = VLContrastModel(
            text_model_name="sentence-transformers/all-mpnet-base-v2",
            vision_model_name="facebook/dinov2-base",
            device=device,
            linear=False,
        )
        # weights_path='/home/mila/l/le.zhang/scratch/light_align/output/raw_data_linear_shared_head/checkpoint_22.pth'
    # checkpoint = torch.load(weights_path)
    # model.vlhead.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    tokenizer = model.text_model.tokenizer
    processor = Processor(model.vision_model.image_processor)
    zeroshot_weights = zeroshot_classifier(IMAGENET_CLASSES, IMAGENET_TEMPLATES)
    images = ImageNetV2Dataset(
        variant="matched-frequency",
        transform=processor,
        location="/home/mila/q/qian.yang/scratch/imagenetv2",
    )

    loader = torch.utils.data.DataLoader(images, batch_size=32, num_workers=2)
    with torch.no_grad():
        top1, top5, n = 0.0, 0.0, 0.0
        for i, (images, target) in enumerate(tqdm(loader)):
            images = images.cuda()
            target = target.cuda()

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
