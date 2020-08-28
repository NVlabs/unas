# -----------------------------------------------------------------
# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# This file has been modified from DARTS library (licensed under
# the Apache License, Version 2.0):
#
# https://github.com/quark0/darts
#
# The Apache License for the original version of this file can be
# found in darts directory. The modifications to this file are subject
# to the same Apache License, Version 2.0.
# -----------------------------------------------------------------

"""Train found model on ImageNet dataset."""

import argparse
import logging
import os
import sys
sys.path.append('../')


from apex.parallel import DistributedDataParallel as DDP
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.utils
import torchvision.datasets as dset
import torchvision.transforms as transforms
from util.datasets import imagenet_lmdb_dataset
from util import utils

from proxylessnas.model import Network
from proxylessnas.compute_flops import find_flops
from proxylessnas import genotypes


parser = argparse.ArgumentParser('imagenet')
# Dataset choices.
parser.add_argument('--data', type=str,
                    default='/tmp/data/',
                    help='location of the data corpus')
# Architecture choices.
parser.add_argument('--genotype', type=str, default='UNAS',
                    help='which architecture to use')
parser.add_argument('--max_M_flops', type=int, default=600,
                    help='max allowed M flops')
parser.add_argument('--layers', type=int, default=21,
                    help='total number of layers')
parser.add_argument('--width_multiplier', type=float, default=1.0,
                    help='width multiplier')
# Optimization choices.
parser.add_argument('--batch_size', type=int, default=64,
                    help='batch size')
parser.add_argument('--checkpoint', type=str, default='',
                    help='path to the model.')

# Logging choices.
parser.add_argument('--report_freq', type=float, default=50,
                    help='report frequency')
parser.add_argument('--save', type=str, default='EXP',
                    help='experiment name')
parser.add_argument('--root_dir', type=str, default='/tmp/',
                    help='root directory')
# Misc.
parser.add_argument('--seed', type=int, default=0,
                    help='random seed')
# DDP.
parser.add_argument('--local_rank', type=int, default=0,
                    help='rank of process')

args = parser.parse_args()

# Set up DDP.
args.distributed = True
torch.cuda.set_device(args.local_rank)
torch.distributed.init_process_group(backend='nccl', init_method='env://')
args.world_size = torch.distributed.get_world_size()


# Set up logging.
assert args.root_dir
args.save = args.root_dir + '/eval_imagenet-{}'.format(args.save)
if args.local_rank == 0:
    utils.create_exp_dir(args.save)
logging = utils.Logger(args.local_rank, args.save)
writer = utils.Writer(args.local_rank, args.save)

CLASSES = 1000


def main():
    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)
    np.random.seed(args.seed)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled = True
    torch.cuda.manual_seed(args.seed)
    logging.info('args = %s', args)

    # Get data loaders.
    traindir = os.path.join(args.data, 'train')
    validdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(
            brightness=0.4,
            contrast=0.4,
            saturation=0.4,
            hue=0.2),
        transforms.ToTensor(),
        normalize,
    ])
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])

    if 'lmdb' in args.data:
        train_data = imagenet_lmdb_dataset(traindir, transform=train_transform)
        valid_data = imagenet_lmdb_dataset(validdir, transform=val_transform)
    else:
        train_data = dset.ImageFolder(traindir, transform=train_transform)
        valid_data = dset.ImageFolder(validdir, transform=val_transform)

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)
    train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, shuffle=False,
        pin_memory=True, num_workers=8, sampler=train_sampler)

    valid_queue = torch.utils.data.DataLoader(
        valid_data, batch_size=args.batch_size, shuffle=False,
        pin_memory=True, num_workers=8)

    # Set up the network.
    if os.path.isfile(args.genotype):
        logging.info('Loading genotype from: %s' % args.genotype)
        genotype = torch.load(args.genotype, map_location='cpu')
        logging.info(genotype)
    else:
        genotype = eval('genotypes.%s' % args.genotype)
        logging.info(genotype)

    flops, num_params = find_flops(genotype, args.layers, args.width_multiplier)
    logging.info('Num flops = %.2fM', flops / 1e6)
    logging.info('Num params = %.2fM', num_params / 1e6)

    logging.info('width multiplier = %f', args.width_multiplier)

    # Create model and loss.
    model = Network(genotype, num_classes=1000, layers=args.layers, width_multiplier=args.width_multiplier)
    model = model.cuda()
    model = DDP(model, delay_allreduce=True)

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()
    logging.info('param size = %fM', utils.count_parameters_in_M(model))

    # load model
    logging.info('loading from checkpoint: \n %s', args.checkpoint)
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'])

    # Validation.
    valid_acc_top1, valid_acc_top5, valid_obj = infer(
        valid_queue, model, criterion)
    logging.info('valid_acc_top1 %f', valid_acc_top1)
    logging.info('valid_acc_top5 %f', valid_acc_top5)


def infer(valid_queue, model, criterion):
    objs = utils.AverageMeter()
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()

    model.eval()
    with torch.no_grad():
        for step, (data, target) in enumerate(valid_queue):
            data = data.cuda()
            target = target.cuda()

            logits = model(data)
            loss = criterion(logits, target)

            prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
            n = data.size(0)
            objs.update(loss.item(), n)
            top1.update(prec1.item(), n)
            top5.update(prec5.item(), n)

            if step % args.report_freq == 0:
                logging.info('valid %03d %e %f %f', step,
                             objs.avg, top1.avg, top5.avg)

    return top1.avg, top5.avg, objs.avg

if __name__ == '__main__':
    main()
