import torch
from torch.utils.data import DataLoader

from dataset.dataset import PretrainDataset, evaluate_dataset, CsvDataset, CsvDataset_no_norm

def create_CSV_dataset(dataset, config):
    if dataset == 'pretrain':
        dataset = CsvDataset(config['train_file'], config)
        return dataset
    
    elif dataset == 'evaluate':
        dataset = CsvDataset(config['test_file'], config)
        return dataset
    
def create_CSV_dataset_no_norm(dataset, config):
    if dataset == 'pretrain':
        dataset = CsvDataset_no_norm(config['train_file'], config)
        return dataset
    
    elif dataset == 'evaluate':
        dataset = CsvDataset_no_norm(config['test_file'], config)
        return dataset

def create_dataset(dataset, data):
    if dataset == 'pretrain':
        dataset = PretrainDataset(data)
        return dataset
    
    elif dataset == 'evaluate':
        dataset = evaluate_dataset(data)
        return dataset

def create_sampler(datasets, shuffles, num_tasks, global_rank):
    samplers = []
    for dataset,shuffle in zip(datasets,shuffles):
        sampler = torch.utils.data.DistributedSampler(dataset, num_replicas=num_tasks, rank=global_rank, shuffle=shuffle)
        samplers.append(sampler)
    return samplers


def create_loader(datasets, samplers, batch_size, num_workers, is_trains, collate_fns):
    loaders = []
    for dataset,sampler,bs,n_worker,is_train,collate_fn in zip(datasets,samplers,batch_size,num_workers,is_trains,collate_fns):
        if is_train:
            shuffle = (sampler is None)
            drop_last = True
        else:
            shuffle = False
            drop_last = False
        loader = DataLoader(
            dataset,
            batch_size=bs,
            num_workers=n_worker,
            pin_memory=True,
            sampler=sampler,
            shuffle=shuffle,
            collate_fn=collate_fn,
            drop_last=drop_last,
        )
        loaders.append(loader)
    return loaders
