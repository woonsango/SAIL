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

# def setup_logging():
#     # Set up logging format
#     logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')

#     # Create a logger
#     logger = logging.getLogger()

#     # Suppress lower level logs for non-master processes
#     if dist.is_initialized() and dist.get_rank() != 0:
#         logger.setLevel(logging.WARN)
#     return logger

# def set_seed(seed):
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)

def estimate_num_training_steps(total_samples, batch_size, num_epochs):
    steps_per_epoch = math.ceil(total_samples / batch_size)
    num_training_steps = steps_per_epoch * num_epochs
    return num_training_steps

# def setup_wandb(args):
#     if dist.get_rank() == 0:
#         wandb.init(project="clip_training", config=args)
#         output_path = os.path.join(args.output_dir, args.output_name or wandb.run.id)
#         os.makedirs(output_path, exist_ok=True)
#         return output_path
#     return None

# def setup_distributed():
#     dist.init_process_group(backend='nccl')
#     device = torch.device('cuda', dist.get_rank())
#     return device

# def create_dataloader(dataset, args):
#     sampler = DistributedSampler(dataset, num_replicas=dist.get_world_size(), rank=dist.get_rank())
#     return DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, sampler=sampler, collate_fn=batch_collate_fn), sampler

# def create_model_and_optimizer(args, device):
#     model = VLContrastModel(
#         text_model_name='sentence-transformers/all-mpnet-base-v2',
#         vision_model_name='facebook/dinov2-base',
#         device=device
#     )
#     model.freeze_except_vlhead()
    
#     # Ensure all parameters have uniform requires_grad
#     for param in model.parameters():
#         if not param.requires_grad:
#             param.requires_grad = False  # Explicitly setting for consistency
    
#     model = model.to(device)
    
#     # Using FSDP with use_orig_params=True to avoid requires_grad inconsistency issues
#     model = torch.distributed.fsdp.FullyShardedDataParallel(model, use_orig_params=True)
    
#     optimizer = Lion(model.parameters(), lr=1e-4, weight_decay=1e-7)
#     return model, optimizer

# def create_scheduler(optimizer, total_samples, args):
#     num_training_steps = estimate_num_training_steps(total_samples, args.batch_size, args.num_epochs)
#     scheduler = get_cosine_schedule_with_warmup(
#         optimizer,
#         num_warmup_steps=math.ceil(0.1 * num_training_steps),
#         num_training_steps=num_training_steps
#     )
#     if dist.get_rank() == 0:
#         logging.info(f"Total samples: {total_samples}, Num training steps: {num_training_steps}")
#     return scheduler

# def save_checkpoint(model, optimizer, scheduler, epoch, output_path):
   
#     checkpoint = {
#         'model_state_dict': model.state_dict(),
#         'optimizer_state_dict': optimizer.state_dict(),
#         'scheduler_state_dict': scheduler.state_dict(),
#         'epoch': epoch
#     }
#     torch.save(checkpoint, os.path.join(output_path, f'checkpoint_{epoch}.pth'))

# def process_batch(batch, tokenizer, device, train_data_type):
#     if train_data_type == "embedding":
#         text_embeddings, vision_embeddings = batch
#         text_embeddings = text_embeddings.to(device)
#         vision_embeddings = vision_embeddings.to(device)
#     else:
#         text, vision_embeddings = batch
#         text_embeddings = tokenizer(text, padding=True, truncation=True, return_tensors='pt').to(device)
#         vision_embeddings = {key: value.to(device) for key, value in vision_embeddings.items()}
#     return text_embeddings, vision_embeddings

# def train_one_epoch(model, tokenizer, dataloader, optimizer, scheduler, device, args, epoch, sampler):
#     model.train()
#     sampler.set_epoch(epoch)
#     scaler = GradScaler() 
#     for batch in tqdm(dataloader):
#         text_embeddings, vision_embeddings = process_batch(batch, tokenizer, device, args.train_data_type)

#         with autocast():  # Use autocast for mixed precision
#             norm_vision_embeddings, norm_text_embeddings, logits_per_text = model(vision_embeddings, text_embeddings)
#              # Log the sizes of norm_vision_embeddings and norm_text_embeddings
#             print(f"Rank {dist.get_rank()}: norm_vision_embeddings size: {norm_vision_embeddings.size()}")
#             print(f"Rank {dist.get_rank()}: norm_text_embeddings size: {norm_text_embeddings.size()}")
            
#             all_logits_per_text = gather_embeddings_and_compute_logits(norm_vision_embeddings, norm_text_embeddings, model)
#             # Log the sizes of all_logits_per_text
#             print(f"Rank {dist.get_rank()}: all_logits_per_text size: {all_logits_per_text.size()}")
#             loss = sigclip_loss(all_logits_per_text)
           
#         optimizer.zero_grad()
#         scaler.scale(loss).backward()  # Scale the loss and backward pass
#         scaler.step(optimizer)
#         scaler.update()
#         scheduler.step()

#         if dist.get_rank() == 0:
#             log_metrics(loss, scheduler, model, args, epoch)

# def gather_embeddings_and_compute_logits(norm_vision_embeddings, norm_text_embeddings, model):
#     world_size = dist.get_world_size()
    
#     # Allocate space for gathered tensors
#     gathered_image_embeddings = [torch.zeros_like(norm_vision_embeddings) for _ in range(world_size)]
#     gathered_text_embeddings = [torch.zeros_like(norm_text_embeddings) for _ in range(world_size)]
    
#     # Perform all_gather operations
#     dist.all_gather(gathered_image_embeddings, norm_vision_embeddings)
#     dist.all_gather(gathered_text_embeddings, norm_text_embeddings)
    
#     # Verify tensor sizes are consistent and log details
#     for i, gathered_embeddings in enumerate(gathered_image_embeddings):
#         if gathered_embeddings.size(0) == 0:
#             raise ValueError(f"Rank {i}: gathered_image_embeddings size is zero.")
#         print(f"Rank {dist.get_rank()}: gathered_image_embeddings[{i}] size: {gathered_embeddings.size()}")
    
#     for i, gathered_embeddings in enumerate(gathered_text_embeddings):
#         if gathered_embeddings.size(0) == 0:
#             raise ValueError(f"Rank {i}: gathered_text_embeddings size is zero.")
#         print(f"Rank {dist.get_rank()}: gathered_text_embeddings[{i}] size: {gathered_embeddings.size()}")
    
#     all_image_embeddings = torch.cat(gathered_image_embeddings)
#     all_text_embeddings = torch.cat(gathered_text_embeddings)
    
#     # Log the sizes before matmul
#     print(f"Rank {dist.get_rank()}: all_image_embeddings size: {all_image_embeddings.size()}")
#     print(f"Rank {dist.get_rank()}: all_text_embeddings size: {all_text_embeddings.size()}")

#     # Verify dimensions before matmul
#     if all_image_embeddings.size(1) != all_text_embeddings.size(1):
#         raise ValueError(f"Dimension mismatch: all_image_embeddings size {all_image_embeddings.size()}, all_text_embeddings size {all_text_embeddings.size()}")

#     # Check for zero-size tensors
#     if all_image_embeddings.size(0) == 0 or all_text_embeddings.size(0) == 0:
#         raise ValueError("One of the gathered tensors has zero size after concatenation.")

#     logits = torch.matmul(all_text_embeddings, all_image_embeddings.t()) * model.vlhead.logit_scale.exp() + model.vlhead.logit_bias
#     print(f"Rank {dist.get_rank()}: logits size: {logits.size()}")

#     return logits


# def log_metrics(loss, scheduler, model, args, epoch):
#     logit_bias = model.vlhead.logit_bias.item()
#     logit_scale = model.vlhead.logit_scale.exp().item()
#     wandb.log({"loss": loss.item(), "epoch": epoch, "lr": scheduler.get_last_lr()[0], "batch_size": args.batch_size, "weight_decay": args.weight_decay, "logit_bias": logit_bias, "logit_scale": logit_scale})


# def train(args):

#     device = setup_distributed()
#     logger = setup_logging()  # Setup logging
#     output_path = setup_wandb(args)
#     set_seed(args.local_rank + 1)
    

#     dataset = build_dataset(args)
#     dataloader, sampler = create_dataloader(dataset, args)

#     model, optimizer = create_model_and_optimizer(args, device)
#     scheduler = create_scheduler(optimizer, len(dataset), args)
#     tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-mpnet-base-v2')
#     for epoch in range(args.num_epochs):
#         train_one_epoch(model, tokenizer, dataloader, optimizer, scheduler, device, args, epoch, sampler)

#         if epoch % args.save_n_iter == 0 and dist.get_rank() == 0:
#             save_checkpoint(model, optimizer, scheduler, epoch, output_path)
#     if dist.get_rank() == 0:
#         save_checkpoint(model, optimizer, scheduler, args.num_epochs, output_path)
#         wandb.finish()

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
    
    for epoch in range(args.num_epochs):
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
            wandb.log({"loss": loss.item(), "epoch": epoch, "lr": scheduler.get_last_lr()[0], "batch_size": args.batch_size, "weight_decay": args.weight_decay, "logit_bias": logit_bias, "logit_scale": logit_scale})

        if epoch % args.save_n_iter == 0:
            torch.save(model.state_dict(), os.path.join(output_path,f'model_{epoch}.pth'))
       
    torch.save(model.state_dict(), os.path.join(args.output_dir, 'model_final.pth'))
    wandb.finish()


if __name__ == "__main__":
    args = parse_args() 
    train(args)
    # print(sentence_embeddings)