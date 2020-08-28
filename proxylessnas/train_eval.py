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
from apex.fp16_utils import to_python_float
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
from proxylessnas import genotypes

from proxylessnas.compute_flops import find_max_channels, find_flops


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
parser.add_argument('--learning_rate', type=float, default=0.025,
                    help='init learning rate')
parser.add_argument('--min_learning_rate', type=float, default=0.0,
                    help='min learning rate for cosine')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='momentum')
parser.add_argument('--weight_decay', type=float, default=0.00005,
                    help='weight decay')
parser.add_argument('--epochs', type=int, default=300,
                    help='num of training epochs')
parser.add_argument('--warmup_epochs', type=int, default=5,
                    help='num of training epochs')
parser.add_argument('--grad_clip', type=float,
                    default=5., help='gradient clipping')
parser.add_argument('--label_smooth', type=float,
                    default=0.1, help='label smoothing')
parser.add_argument('--no_scale_lr', action='store_true', default=False,
                    help='use lr scaling based on batch size')
# Logging choices.
parser.add_argument('--report_freq', type=float, default=50,
                    help='report frequency')
parser.add_argument('--save', type=str, default='EXP',
                    help='experiment name')
parser.add_argument('--root_dir', type=str, default='/tmp/checkpoints/',
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


class CrossEntropyLabelSmooth(nn.Module):
    """Smoothed xentropy loss."""

    def __init__(self, num_classes, epsilon):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros_like(log_probs).scatter_(
            1, targets.unsqueeze(1), 1)
        targets = (1 - self.epsilon) * targets + \
            self.epsilon / self.num_classes
        loss = (-targets * log_probs).mean(0).sum()
        return loss


def main():
    # Scale learning rate based on global batch size.
    if not args.no_scale_lr:
        scale = float(args.batch_size * args.world_size) / 64.0
        args.learning_rate = scale * args.learning_rate

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

    # If num channels not provided, find the max under 600M MAdds.
    if args.width_multiplier < 0:
        if args.local_rank == 0:
            flops, num_params, width_multiplier = find_max_channels(
                genotype, args.layers, args.max_M_flops * 1e6)
            logging.info('Num flops = %.2fM', flops/1e6)
            logging.info('Num params = %.2fM', num_params/1e6)
        else:
            width_multiplier = 0
        # All reduce with world_size 1 is sum.
        width_multiplier = torch.Tensor([width_multiplier]).cuda()
        width_multiplier = utils.reduce_tensor(width_multiplier, 1)
        args.width_multiplier = float(width_multiplier.item())
    else:
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
    criterion_smooth = CrossEntropyLabelSmooth(CLASSES, args.label_smooth)
    criterion_smooth = criterion_smooth.cuda()
    logging.info('param size = %fM', utils.count_parameters_in_M(model))

    # Set up network weights optimizer.
    optimizer = torch.optim.SGD(
        model.parameters(),
        args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, float(args.epochs), eta_min=args.min_learning_rate)

    # Train.
    global_step = 0
    best_acc_top1 = 0
    for epoch in range(args.epochs):
        # Shuffle the sampler, update lrs.
        train_queue.sampler.set_epoch(epoch + args.seed)
        # Change lr.
        if epoch >= args.warmup_epochs:
            scheduler.step()
        # model.module.drop_path_prob = args.drop_path_prob * epoch / args.epochs

        # Training.
        train_acc, train_obj, global_step = train(
            train_queue, model, criterion_smooth, optimizer, epoch,
            args.learning_rate, args.warmup_epochs, global_step)
        logging.info('train_acc %f', train_acc)
        writer.add_scalar('train/acc', train_acc, global_step)

        # Validation.
        valid_acc_top1, valid_acc_top5, valid_obj = infer(
            valid_queue, model, criterion)
        logging.info('valid_acc_top1 %f', valid_acc_top1)
        logging.info('valid_acc_top5 %f', valid_acc_top5)
        writer.add_scalar('val/acc_top1', valid_acc_top1, global_step)
        writer.add_scalar('val/acc_top5', valid_acc_top5, global_step)
        writer.add_scalar('val/loss', valid_obj, global_step)

        is_best = False
        if valid_acc_top1 > best_acc_top1:
            best_acc_top1 = valid_acc_top1
            is_best = True

        if args.local_rank == 0:
            utils.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_acc_top1': best_acc_top1,
                'optimizer': optimizer.state_dict(),
            }, is_best, args.save)

    writer.flush()


def train(train_queue, model, criterion, optimizer, epoch, init_lr, warmup_epochs, global_step):
    objs = utils.AverageMeter()
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()

    model.train()
    for step, (data, target) in enumerate(train_queue):
        n = data.size(0)
        data = data.cuda()
        target = target.cuda()

        # Change lr.
        if epoch < warmup_epochs:
            len_epoch = len(train_queue)
            scale = float(1 + step + epoch * len_epoch) / \
                (warmup_epochs * len_epoch)
            lr = init_lr * scale
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        # Forward.
        optimizer.zero_grad()
        logits = model(data)
        loss = criterion(logits, target)

        # Backward and step.
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        # Calculate the accuracy.
        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        reduced_loss = utils.reduce_tensor(loss.data, args.world_size)
        prec1 = utils.reduce_tensor(prec1, args.world_size)
        prec5 = utils.reduce_tensor(prec5, args.world_size)

        objs.update(to_python_float(reduced_loss), n)
        top1.update(to_python_float(prec1), n)
        top5.update(to_python_float(prec5), n)

        if step % args.report_freq == 0:
            current_lr = list(optimizer.param_groups)[0]['lr']
            logging.info('train %03d %e %f %f lr: %e', step,
                         objs.avg, top1.avg, top5.avg, current_lr)
            writer.add_scalar('train/loss', objs.avg, global_step)
            writer.add_scalar('train/acc_top1', top1.avg, global_step)
            writer.add_scalar('train/acc_top5', top5.avg, global_step)
            writer.add_scalar('train/lr', optimizer.state_dict()[
                              'param_groups'][0]['lr'], global_step)
        global_step += 1

    return top1.avg, objs.avg, global_step


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
