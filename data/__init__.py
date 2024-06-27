from .embedding_data import LazyVLEmbeddingDataset
from .vldata import VLDataset, batch_collate_fn

def build_dataset(args):

    if args.data_path is not None and args.image_dir is not None:
        return VLDataset(args.data_path, args.image_dir, transform=None, tokenizer=None)
    
    elif args.text_embedding_dir is not None or args.image_embedding_dir is not None:
        return LazyVLEmbeddingDataset(args.text_embedding_dir, args.image_embedding_dir)
    else:
        raise ValueError("Please provide either json_path and image_dir or text_embedding_dir and image_embedding_dir")