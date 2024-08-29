from tqdm import tqdm
from .utils import get_model_device, save_features, load_features
import os
import torch
from subprocess import call
from .sugar_crepe import SugarCrepe, SugarCrepeFilenames


class Processor:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, images):
        images = self.processor(images, return_tensors="pt")["pixel_values"][0]
        return images


def SugarCrepe_eval_task(
    model,
    bs,
    images_dir,
    ann_file,
    pre_encode_image_features,
    pre_encode_text_features,
    device,
):
    image_score = []
    text_score = []
    score = []
    dataset = SugarCrepeFilenames(root=images_dir, ann_file=ann_file)

    filename_loader = torch.utils.data.DataLoader(
        dataset, batch_size=bs, num_workers=2, shuffle=False
    )
    for image_name, ann_idx in tqdm(filename_loader):
        batch_images = []
        batch_texts = []
        for j, idx in enumerate(ann_idx):
            batch_images.append(pre_encode_image_features[image_name[j]])
            batch_texts.append(torch.stack(pre_encode_text_features[int(idx)]))
        batch_images = torch.stack(batch_images)
        batch_texts = torch.stack(batch_texts)
        if len(batch_images.shape) == 2:
            B, dim = batch_images.shape
            batch_images = batch_images.view(B, 1, dim)
        # batch_images: B, nb_images_per_instance, C, H, W
        # batch_texts: B, nb_captions_per_instance
        B, nim, dim = batch_images.shape
        nt = len(batch_texts[0])
        batch_texts_ = batch_texts.view(B * nt, -1).to(device)
        batch_images = batch_images.to(device)
        batch_images_ = batch_images.view(B * nim, dim)  # B*nim, C, H, W
        # compute the embedding of images and texts
        with torch.no_grad():
            batch_images_emb = model.encode_image(
                batch_images_, is_pre_encoded=True
            ).view(B, nim, -1)
            batch_texts_emb = model.encode_text(batch_texts_, is_pre_encoded=True).view(
                B, nt, -1
            )
        gt = torch.arange(min(nim, nt)).to(device)
        for i in range(B):
            # iteratve over instances

            # compute similarities between each image and each text
            images_emb = batch_images_emb[i]
            texts_emb = batch_texts_emb[i]
            scores = images_emb @ texts_emb.t()

            image_closest_text = scores.argmax(dim=1)[: len(gt)]
            pred_text_is_correct = (image_closest_text == gt).all().item()
            text_score.append(pred_text_is_correct)
    return torch.Tensor(text_score).float().mean().item()


def SugarCrepe_eval(
    model, text_model_name, vision_model_name, images_dir, save_dir, bs=1024
):
    device = get_model_device(model)
    task_list = [
        "add_att",
        "add_obj",
        "replace_att",
        "replace_obj",
        "replace_rel",
        "swap_att",
        "swap_obj",
    ]
    processor = Processor(model.vision_model.image_processor)
    task_scores = {}
    for task in task_list:
        ann = f"{images_dir}/{task}.json"
        if not os.path.exists(ann):
            url = f"https://raw.githubusercontent.com/RAIVNLab/sugar-crepe/main/data/{task}.json"
            call(f"wget {url} --output-document={ann}", shell=True)
        dataset = SugarCrepe(images_dir, ann, transform=processor)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=bs, num_workers=2, shuffle=False
        )

        text_feature_path = f"{save_dir}/{text_model_name}/SugarCrepe_{task}.pt"
        image_feature_path = f"{save_dir}/{vision_model_name}/SugarCrepe_{task}.pt"
        if not os.path.exists(text_feature_path) or not os.path.exists(
            image_feature_path
        ):
            print(f"Extracting backbone features for {task}")
            pre_encode_text_features = {}
            pre_encode_image_features = {}
            with torch.no_grad():
                for batch_images, batch_texts, image_name, ann_idx in tqdm(dataloader):

                    batch_texts = model.text_model.tokenizer(
                        [text for i, texts in enumerate(batch_texts) for text in texts],
                        padding=True,
                        truncation=True,
                        return_tensors="pt",
                    ).to(device)
                    class_embeddings, class_features = model.encode_text(
                        batch_texts, return_encoded=True
                    )
                    for j, idx in enumerate(ann_idx):
                        # each image has two captions
                        pre_encode_text_features[int(idx)] = [
                            class_features[j * 2].cpu(),
                            class_features[j * 2 + 1].cpu(),
                        ]
                    batch_images = batch_images.to(device)
                    image_features, encoded_features = model.encode_image(
                        {"pixel_values": batch_images}, return_encoded=True
                    )
                    for j, name in enumerate(image_name):
                        pre_encode_image_features[name] = encoded_features[j].cpu()
            save_features(pre_encode_text_features, text_feature_path)
            save_features(pre_encode_image_features, image_feature_path)
        else:
            print(f"Loading backbone text features from {text_feature_path}")
            pre_encode_text_features = load_features(text_feature_path)
            print(f"Loading backbone image features from {image_feature_path}")
            pre_encode_image_features = load_features(image_feature_path)
        eval_score = SugarCrepe_eval_task(
            model,
            bs,
            images_dir,
            ann,
            pre_encode_image_features,
            pre_encode_text_features,
            device,
        )
        task_scores[task] = eval_score

    return task_scores
