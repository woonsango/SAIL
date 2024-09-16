from .embedding_data import VLEmbeddingDataset, custom_collate_fn
from torch.utils.data.distributed import DistributedSampler
from dataclasses import dataclass
from multiprocessing import Value
from torch.utils.data import DataLoader

class SharedEpoch:
    def __init__(self, epoch: int = 0):
        self.shared_epoch = Value('i', epoch)

    def set_value(self, epoch):
        self.shared_epoch.value = epoch

    def get_value(self):
        return self.shared_epoch.value

@dataclass
class DataInfo:
    dataloader: DataLoader
    sampler: DistributedSampler = None
    shared_epoch: SharedEpoch = None
    data_info: dict = None

    def set_epoch(self, epoch):
        if self.shared_epoch is not None:
            self.shared_epoch.set_value(epoch)
        if self.sampler is not None and isinstance(self.sampler, DistributedSampler):
            self.sampler.set_epoch(epoch)

def get_embedding_dataset(args, is_train, epoch=0):
    input_filename = args.text_embedding_list and args.image_embedding_list
    assert input_filename, "Please provide text_embedding_list and image_embedding_list"
    dataset = VLEmbeddingDataset(
        args.text_embedding_list,
        args.image_embedding_list,
        args.train_num_samples
    )
    num_samples = len(dataset)
    sampler = DistributedSampler(dataset) if args.distributed and is_train else None
    shuffle = is_train and sampler is None
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        collate_fn=custom_collate_fn,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=is_train,
    )
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)


    return DataInfo(dataloader, sampler, data_info={'num_samples': num_samples, 'visual_dim': dataset.visual_dim, 'text_dim': dataset.text_dim})

def get_data(args, epoch=0):
    data = {}
    if args.text_embedding_list and args.image_embedding_list:
        data['train'] = get_embedding_dataset(args, is_train=True, epoch=epoch)
    else:
        raise ValueError(f"Unknown dataset type: {args.dataset_type}")
    return data


