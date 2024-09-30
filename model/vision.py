from transformers import AutoImageProcessor, AutoModel
import torch
import torch.nn as nn
from PIL import Image, ImageFile
import os
from typing import List, Tuple, Dict, Any, Union, Optional

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

class ImageEmbedding(nn.Module):
    def __init__(self, model_name="facebook/dinov2-base", device=None, test: bool = False):
        super(ImageEmbedding, self).__init__()
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        
        if any(x in model_name.lower() for x in ['mae', 'dinov2']) and not test:
            print("Using model with SDPA")
            self.SDPA = True
            self.model = AutoModel.from_pretrained(model_name, attn_implementation="sdpa", torch_dtype=torch.float16).to(self.device)
            self.image_processor = AutoImageProcessor.from_pretrained(model_name)
        else:
            self.SDPA = False
            self.model = AutoModel.from_pretrained(model_name, torch_dtype=torch.float16).to(self.device)
            self.image_processor = AutoImageProcessor.from_pretrained(model_name)

    def load_images_from_directory(self, images_path: List[str]) -> List[Image.Image]:
        images = []
        for image_path in images_path:
            with Image.open(image_path) as img:
                img = img.convert("RGB")
                images.append(img)
        return images

    def get_visual_embeddings_from_directory(self, images_path: List[str]):
        images = self.load_images_from_directory(images_path)
        inputs = self.image_processor(images, return_tensors="pt").to(self.model.device, dtype=self.model.dtype)
        with torch.no_grad():
            return self.forward(inputs)

    def forward(self, inputs):
        outputs = self.model(**inputs)
        sequence_output = outputs[0]  # batch_size, sequence_length, hidden_size
        if 'mae' in self.model.config._name_or_path.lower():
            linear_input = sequence_output.mean(dim=1)
        else:
            cls_token = sequence_output[:, 0]
            patch_tokens = sequence_output[:, 1:]
            linear_input = torch.cat([cls_token, patch_tokens.mean(dim=1)], dim=1)
        return linear_input

    def patch_embeddings(self, inputs,skip_res=False):
        if skip_res:
            outputs = self.model.skip_res(**inputs)
        else:
            outputs = self.model(**inputs)
        sequence_output = outputs[0] 
        patch_tokens = sequence_output[:, 1:]
        cls_token = sequence_output[:, 0].unsqueeze(1).repeat(1, patch_tokens.shape[1], 1)
        linear_input = torch.cat([cls_token, patch_tokens], dim=-1)
        return linear_input


if __name__ == "__main__":
    processor = ImageEmbedding()
    breakpoint()
    image_dirs = ["path/to/dir1", "path/to/dir2"]
    hidden_states = processor.get_visual_embeddings_from_directory(image_dirs)
    for states in hidden_states:
        print(list(states.shape))
