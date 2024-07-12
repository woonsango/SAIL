from tqdm import tqdm
from datasets import load_dataset
from .utils import get_model_device
import os
import torch

def text_correct(result):
    return result["c0_i0"] > result["c1_i0"] and result["c1_i1"] > result["c0_i1"]

def image_correct(result):
    return result["c0_i0"] > result["c0_i1"] and result["c1_i1"] > result["c1_i0"]

def group_correct(result):
    return image_correct(result) and text_correct(result)

def winoground_eval(model):
    auth_token = os.getenv("HF_AUTH_TOKEN")
    device = get_model_device(model)
    winoground = load_dataset("facebook/winoground", use_auth_token=auth_token)["test"]
    winoground_clip_scores = []
    for example in tqdm(winoground):
        text = [example["caption_0"],example["caption_1"]]
        text = model.text_model.tokenizer(text, padding=True, truncation=True, return_tensors='pt').to(device)
        image0 = example["image_0"].convert("RGB")
        image1 = example["image_1"].convert("RGB")
        images = model.vision_model.image_processor([image0,image1], return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(images, text)

        logits_per_text = outputs["logits_per_text"]
        clip_score_c0_i0 = logits_per_text[0,0].item()
        clip_score_c1_i0 = logits_per_text[1,0].item()
        clip_score_c0_i1 = logits_per_text[0,1].item()
        clip_score_c1_i1 = logits_per_text[1,1].item()
        winoground_clip_scores.append({"id" : example["id"], "c0_i0": clip_score_c0_i0, "c0_i1": clip_score_c0_i1, "c1_i0": clip_score_c1_i0, "c1_i1": clip_score_c1_i1})

    text_correct_count = 0
    image_correct_count = 0
    group_correct_count = 0
    for result in winoground_clip_scores:
        text_correct_count += 1 if text_correct(result) else 0
        image_correct_count += 1 if image_correct(result) else 0
        group_correct_count += 1 if group_correct(result) else 0

    denominator = len(winoground_clip_scores)
    print("text score:", text_correct_count/denominator)
    print("image score:", image_correct_count/denominator)
    print("group score:", group_correct_count/denominator)
    return {"text": text_correct_count/denominator, "image": image_correct_count/denominator, "group": group_correct_count/denominator}