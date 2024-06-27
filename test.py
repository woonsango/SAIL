from loss import clip_loss, sigclip_loss
import torch
from model import VLContrastModel
from PIL import Image
from datasets import load_dataset

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = VLContrastModel(text_model_name='sentence-transformers/all-mpnet-base-v2', vision_model_name='facebook/dinov2-base', device=device, linear=False)
    weights_path='/home/mila/l/le.zhang/scratch/light_align/output/raw_data/model_20.pth'
    checkpoint = torch.load(weights_path, map_location=device)
    model.load_state_dict(checkpoint)
    model = model.to(device)
    model.eval()
    text = ['a photo of a cat', 'a photo of a dog']
    text = model.text_model.tokenizer(text, padding=True, truncation=True, return_tensors='pt').to(device)
    
    image0 = Image.open('/home/mila/l/le.zhang/scratch/light_align/cat.jpeg')
    image1 = Image.open('/home/mila/l/le.zhang/scratch/light_align/dog.jpeg')
    images = model.vision_model.image_processor([image0,image1], return_tensors="pt").to(device)
    with torch.no_grad():
        _, _,logits = model(images, text)
    print(logits)