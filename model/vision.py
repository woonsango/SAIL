from transformers import AutoImageProcessor, Dinov2Model
import torch
import torch.nn as nn
from PIL import Image
import os
from typing import List, Tuple, Dict, Any, Union, Optional

class ImageEmbedding(nn.Module):
    def __init__(self, model_name="facebook/dinov2-base", device=None):
        super(ImageEmbedding, self).__init__()
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.image_processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = Dinov2Model.from_pretrained(model_name).to(self.device)


    def load_images_from_directory(self, images_path: List[str]) -> List[Image.Image]:
        images = []
        for image_path in images_path:
            with Image.open(image_path) as img:
                img = img.convert("RGB")
                images.append(img.copy())
        return images
    
    def get_visual_embeddings_from_directory(self, images_path: List[str]):
        images = self.load_images_from_directory(images_path)
        inputs = self.image_processor(images, return_tensors="pt").to(self.device)
        with torch.no_grad():
            return self.forward(inputs)

    def forward(self, inputs):
        outputs = self.model(**inputs)
        sequence_output = outputs[0]  # batch_size, sequence_length, hidden_size
        cls_token = sequence_output[:, 0]
        patch_tokens = sequence_output[:, 1:]
        linear_input = torch.cat([cls_token, patch_tokens.mean(dim=1)], dim=1)
        return linear_input

if __name__ == "__main__":
    processor = ImageEmbedding()
    breakpoint()
    image_dirs = ["path/to/dir1", "path/to/dir2"]
    hidden_states = processor.get_visual_embeddings_from_directory(image_dirs)
    for states in hidden_states:
        print(list(states.shape))
