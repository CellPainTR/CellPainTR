import argparse
import os
import ruamel.yaml as yaml
import numpy as np
import random
import time
import datetime
import json
from pathlib import Path

import torch
import torch.nn as nn #grandient norm
import torch.backends.cudnn as cudnn
import torch.distributed as dist

from models.model_pretrain import HyenaMorph
from models.tokenization import MorphTokenizer

import utils
from dataset import create_dataset, create_sampler, create_loader, create_CSV_dataset
from scheduler import create_scheduler
from optim import create_optimizer


def train(model, data_loader, samplers, optimizer, tokenizer, epoch, warmup_steps, device, scheduler, config, args):
    
    def log_tensor_stats(tensor, name):
        print(f"{name} stats: mean={tensor.mean().item():.4f}, min={tensor.min().item():.4f}, max={tensor.max().item():.4f}, std={tensor.std().item():.4f}")
        
    model.train()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=50, fmt='{value:.6f}'))
    metric_logger.add_meter('loss', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))

    header = 'Train Epoch: [{}]'.format(epoch)
    print_freq = 1
    step_size = 100
    warmup_iterations = warmup_steps * step_size

    if args.distributed:
        data_loader.sampler.set_epoch(epoch)

    for k, data in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        print(f"Current File: {data['file_name']}")
        datasets = [create_dataset('pretrain', data)]
        data_loader_rows = create_loader(datasets, samplers, batch_size=[config['batch_size']], num_workers=[4], is_trains=[True], collate_fns=[None])[0]
        print(f"Current Plate sample: {len(data['Pheno_seq'][0])}")
        for i, subdata in enumerate(data_loader_rows, print_freq):
            optimizer.zero_grad()
            
            Morph_ids = subdata['Pheno_ids']
            Feature_seq = subdata['Pheno_seq']
            Pheno_group_mask = subdata['Pheno_group_mask']
            Source_num = subdata['source_num']
            
            Morph_input = tokenizer(Morph_ids, padding='longest', return_tensors="pt").to(device)
            Feature_seq = Feature_seq.to(device)
            Pheno_group_mask = Pheno_group_mask.to(device)
            Source_num = Source_num.to(device)
    
            if epoch > 0:
                alpha = config['alpha']
            else:
                alpha = config['alpha'] * min(1, i / len(data_loader_rows))
                    
            loss = model(Morph_input, Feature_seq, Pheno_group_mask, Source_num, alpha=alpha)
            
            # Add NaN check for loss
            if torch.isnan(loss):
                print("NaN loss encountered. Skipping batch.")
                continue
            
            if loss > 1e6:  # Adjust threshold as needed
                torch.save({
                    'Morph_input': Morph_input,
                    'Feature_seq': Feature_seq,
                    'Pheno_group_mask': Pheno_group_mask,
                    'Source_num': Source_num,
                    'model_output': loss,
                    'loss': loss
                }, f'debug_batch_{epoch}_{i}.pt')
                print(f"Saved problematic batch to debug_batch_{epoch}_{i}.pt")

            loss.backward()

            # Add NaN check for gradients
            if any(torch.isnan(p.grad).any() for p in model.parameters() if p.grad is not None):
                print("NaN gradient encountered. Skipping batch.")
                optimizer.zero_grad()
                continue
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config['max_grad_norm'])
            
            # Compute gradient norm
            total_norm = 0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5
            
            # Log the gradient norm
            metric_logger.update(grad_norm=total_norm)
            
            optimizer.step()
    
            metric_logger.update(loss=loss.item())
            metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    
            if epoch == 0 and i % step_size == 0 and i <= warmup_iterations:
                scheduler.step(i // step_size)

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())
    return {k: "{:.3f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}


def main(args, config):
    utils.init_distributed_mode(args)

    device = torch.device(args.device)

    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    start_epoch = 0
    max_epoch = config['schedular']['epochs']
    warmup_steps = config['schedular']['warmup_epochs']

    print("Creating dataset")
    datasets = [create_CSV_dataset('pretrain', config)]
    
    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        samplers = create_sampler(datasets, [True], num_tasks, global_rank)
    else:
        samplers = [None]
        
    data_loader = create_loader(datasets, samplers, batch_size=[1], num_workers=[4], is_trains=[True], collate_fns=[None])[0]
    
    tokenizer = MorphTokenizer(vocab_file=os.path.join(config['data_path'], config['vocab']))
   
    print("Creating model")
    model = HyenaMorph(tokenizer=tokenizer, config=config)

    model = model.to(device)

    arg_opt = utils.AttrDict(config['optimizer'])
    optimizer = create_optimizer(arg_opt, model)
    arg_sche = utils.AttrDict(config['schedular'])
    lr_scheduler, _ = create_scheduler(arg_sche, optimizer)

    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        state_dict = checkpoint['model']
        if args.resume:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            start_epoch = checkpoint['epoch'] + 1
        model.load_state_dict(state_dict)
        print('load checkpoint from %s' % args.checkpoint)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
    
    print("Start training")
    start_time = time.time()

    for epoch in range(start_epoch, max_epoch):

        if epoch > 0:
            lr_scheduler.step(epoch + warmup_steps)

        train_stats = train(model, data_loader, samplers, optimizer, tokenizer, epoch, warmup_steps, device, lr_scheduler, config, args)
        if utils.is_main_process():
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         'epoch': epoch,
                         }
            save_obj = {
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'config': config,
                'epoch': epoch,
            }
            torch.save(save_obj, os.path.join(args.output_dir, 'checkpoint_%02d.pth' % epoch))

            with open(os.path.join(args.output_dir, "log.txt"), "a") as f:
                f.write(json.dumps(log_stats) + "\n")

        #dist.barrier()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/Pretrain.yaml')
    parser.add_argument('--checkpoint', default='...')
    parser.add_argument('--resume', default=True, type=bool)
    parser.add_argument('--output_dir', default='...')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', default=False, type=bool)
    args = parser.parse_args()

    yaml = yaml.YAML(typ='rt')
    config = yaml.load(open(args.config, 'r'))

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))

    main(args, config)