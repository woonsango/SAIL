import json
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch
from transformers import AutoImageProcessor

class VLDataset(Dataset):
    def __init__(self, json_path, image_dir, transform=None, tokenizer=None):
        """
        Args:
            json_path (str): Path to the JSON file.
            image_dir (str): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        with open(json_path, 'r') as f:
            self.data = json.load(f)

        if transform is None:
            self.transform = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
        else:
            self.transform = transform  
            
        self.image_dir = image_dir
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        img_name = os.path.join(self.image_dir, self.data[idx]['image'])
        image = Image.open(img_name).convert('RGB')
        image = self.transform(image, return_tensors="pt")['pixel_values']
        text = self.data[idx]['conversations'][1]['value']
        return text, image
    
def batch_collate_fn(batch):
    text_list = []
    image_list = []

    for item in batch:
        text, image = item
        text_list.append(text)
        image_list.append(image[0])
    images = torch.stack(image_list)
    images = {'pixel_values': images}
    return text_list, images



if __name__ == "__main__":


    # Example usage:
    json_path = '/home/mila/l/le.zhang/scratch/light_align/data/blip_laion_cc_sbu_558k.json'
    image_dir = '/home/mila/l/le.zhang/scratch/light_align/data/image'
    image_processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
    # Create the dataset
    dataset = VLDataset(json_path=json_path, image_dir=image_dir, transform=image_processor)
    dataloader = DataLoader(dataset, collate_fn=batch_collate_fn, batch_size=4, shuffle=True, num_workers=4)
    for text, image in dataloader:
        breakpoint()
        # print(text, image)
        break
