import argparse
import json
import os
import torch
from tqdm import tqdm
from model import VLContrastHead, VLContrastModel
from data import build_dataset, batch_collate_fn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from loss import sigclip_loss
from optimizer import Lion
import wandb
from transformers import get_cosine_schedule_with_warmup, AutoTokenizer
import math
import logging

from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
import random
import numpy as np
from torch.cuda.amp import GradScaler, autocast

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')

# Create a logger
logger = logging.getLogger()

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training")
    # embedding data
    parser.add_argument("--text_embedding_dir", type=str, help="directory containing text embeddings")
    parser.add_argument("--image_embedding_dir", type=str, help="directory containing image embeddings")

    # raw text image data
    parser.add_argument("--data_path", type=str, help="path to json file containing text and image paths")
    parser.add_argument("--image_dir", type=str, help="directory containing image directory")


    parser.add_argument("--batch_size", type=int, default=128, help="training batch size") 
    parser.add_argument("--num_workers", type=int, default=4, help="number of workers for dataloader")   
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--num_epochs", type=int, default=10, help="number of training num_epochs")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="weight decay for regularization")
    parser.add_argument("--save_n_iter", type=int, default=1, help="save model every n iterations")
    parser.add_argument("--output_dir", type=str, default= './output' ,help="output directory")
    parser.add_argument("--output_name", type=str, default= None ,help="output directory")

    args = parser.parse_args()
    assert (args.text_embedding_dir and args.image_embedding_dir) or (args.data_path and args.image_dir), "Please provide either text_embedding_dir and image_embedding_dir or data_path and image_dir"

    # but not both set
    assert not (args.text_embedding_dir and args.data_path), "Please provide either text_embedding_dir and image_embedding_dir or data_path and image_dir"

    if args.text_embedding_dir:
        args.train_data_type = "embedding"
    else:
        args.train_data_type = "image_text"

    return args

def estimate_num_training_steps(total_samples, batch_size, num_epochs):
    steps_per_epoch = math.ceil(total_samples / batch_size)
    num_training_steps = steps_per_epoch * num_epochs
    return num_training_steps

def train(args):
    wandb.init(project="clip_training", config=args)

    if args.output_name:
        output_path = os.path.join(args.output_dir, args.output_name)
    else:
        cur_time = wandb.run.id
        output_path = os.path.join(args.output_dir, cur_time)
    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)

    dataset = build_dataset(args)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if args.train_data_type == "embedding":
        model = VLContrastHead(vision_dimesion=1536, text_dimension=768, device=device)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    else:
        model = VLContrastModel(text_model_name='sentence-transformers/all-mpnet-base-v2', vision_model_name='facebook/dinov2-base', device=device)
        model.freeze_except_vlhead()
        dataloader = DataLoader(dataset, collate_fn=batch_collate_fn, batch_size=args.batch_size, shuffle=True, num_workers=4)


    tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-mpnet-base-v2')

     # Load the latest checkpoint if available
    start_epoch = 0
    checkpoint_files = [f for f in os.listdir(output_path) if f.startswith('checkpoint_') and f.endswith('.pth')]
    if checkpoint_files:
        
        latest_checkpoint = max(checkpoint_files, key=lambda x: int(x.split('_')[1].split('.')[0]))
        logging.info(f"Loading checkpoint for VL head: {latest_checkpoint}")
        checkpoint_path = os.path.join(output_path, latest_checkpoint)
        checkpoint = torch.load(checkpoint_path)
        model.vlhead.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1

    model.train()
    model = model.to(device)
    optimizer = Lion(model.parameters(), lr=1e-4, weight_decay=1e-7)
    total_samples = len(dataset)
    num_training_steps = estimate_num_training_steps(total_samples, args.batch_size, args.num_epochs)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=math.ceil(0.1*num_training_steps),
        num_training_steps=num_training_steps
    )
    logging.info(f"Total samples: {total_samples}, Num training steps: {num_training_steps}")
    
    for epoch in range(start_epoch, args.num_epochs):
        for batch in tqdm(dataloader):
            if args.train_data_type == "embedding":
                text_embeddings, vision_embeddings = batch
                text_embeddings = text_embeddings.to(device)
                vision_embeddings = vision_embeddings.to(device)
            else:
                text, vision_embeddings = batch
                text_embeddings = tokenizer(text, padding=True, truncation=True, return_tensors='pt').to(device)
                vision_embeddings = {key: value.to(device) for key, value in vision_embeddings.items()}
                
                
            _, _, logits_per_text = model(vision_embeddings, text_embeddings)
            loss = sigclip_loss(logits_per_text)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            logit_bias = model.logit_bias.item() if args.train_data_type == "embedding" else model.vlhead.logit_bias.item()
            logit_scale = model.logit_scale.exp().item() if args.train_data_type == "embedding" else model.vlhead.logit_scale.exp().item()
            logging.info(f"Epoch: {epoch}, Loss: {loss.item()}, LR: {scheduler.get_last_lr()[0]}, Batch size: {args.batch_size}, Weight decay: {args.weight_decay}, Logit bias: {logit_bias}, Logit scale: {logit_scale}")
            wandb.log({
                "loss": loss.item(),
                "epoch": epoch,
                "lr": scheduler.get_last_lr()[0],
                "batch_size": args.batch_size,
                "weight_decay": args.weight_decay,
                "logit_bias": logit_bias,
                "logit_scale": logit_scale
            })

        if epoch % args.save_n_iter == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.vlhead.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            }, os.path.join(output_path, f'checkpoint_{epoch}.pth'))
       
    torch.save(model.vlhead.state_dict(), os.path.join(args.output_dir, 'model_final.pth'))
    wandb.finish()


if __name__ == "__main__":
    args = parse_args() 
    train(args)
    # print(sentence_embeddings)