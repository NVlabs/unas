# -----------------------------------------------------------------
# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# This file has been modified from DARTS library (licensed under
# the Apache License, Version 2.0):
#
# https://github.com/quark0/darts
#
# The Apache License for the original version of this file can be
# found in this directory. The modifications to this file are subject
# to the same Apache License, Version 2.0.
# -----------------------------------------------------------------

"""Train found model on datasets."""

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

from util import datasets
from darts import genotypes

parser = argparse.ArgumentParser('Train')
# Dataset choices.
parser.add_argument('--data', type=str, default='/tmp/data/',
                    help='location of the data corpus')
parser.add_argument('--dataset', type=str, default='cifar10',
                    choices=['cifar10', 'cifar100'],
                    help='which dataset to use')
# Architecture choices.
parser.add_argument('--genotype', type=str, default='UNAS_CIFAR10',
                    help='which architecture to use')
parser.add_argument('--init_channels', type=int, default=36,
                    help='num of init channels')
parser.add_argument('--layers', type=int, default=20,
                    help='total number of layers')
parser.add_argument('--auxiliary', action='store_true', default=True,
                    help='use auxiliary tower')
parser.add_argument('--auxiliary_weight', type=float, default=0.4,
                    help='weight for auxiliary loss')
# Optimization choices.
parser.add_argument('--batch_size', type=int, default=96,
                    help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.025,
                    help='init learning rate')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4,
                    help='weight decay')
parser.add_argument('--grad_clip', type=float, default=5,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=600,
                    help='num of training epochs')
parser.add_argument('--warmup_epochs', type=int, default=0,
                    help='num of warming up training epochs')
parser.add_argument('--cutout', action='store_true', default=True,
                    help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16,
                    help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.2,
                    help='drop path probability')
# Logging choices.
parser.add_argument('--report_freq', type=float, default=50,
                    help='report frequency')
parser.add_argument('--model_path', type=str, default='saved_models',
                    help='path to save the model')
parser.add_argument('--save', type=str, default='EXP',
                    help='experiment name')
parser.add_argument('--root_dir', type=str, default='/tmp/results/',
                    help='root directory')
# Misc.
parser.add_argument('--seed', type=int, default=0,
                    help='random seed')
parser.add_argument('--gpu', type=int, default=0,
                    help='gpu device id')
# DDP.
parser.add_argument('--local_rank', type=int, default=0,
                    help='rank of process')

args = parser.parse_args()

from darts.genotypes import set_primitives
set_primitives(-1)
from util import utils
from darts.model import Network

# Set up DDP.
args.distributed = True
torch.cuda.set_device(args.local_rank)
torch.distributed.init_process_group(backend='nccl', init_method='env://')
args.world_size = torch.distributed.get_world_size()

# Set up logging.
assert args.root_dir
args.save = args.root_dir + '/eval-{}'.format(args.save)
if args.local_rank == 0:
    utils.create_exp_dir(args.save)
logging = utils.Logger(args.local_rank, args.save)
writer = utils.Writer(args.local_rank, args.save)


def main():
    """Do everything!"""
    if args.world_size > 1:
        # Scale learning rate based on global batch size.
        scale = float(args.batch_size * args.world_size) / 64.0
        args.learning_rate = scale * args.learning_rate

    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    cudnn.benchmark = True
    cudnn.enabled = True
    torch.cuda.manual_seed(args.seed)
    logging.info('gpu device = %d' % args.gpu)
    logging.info('args = %s', args)

    # Get data loaders.
    train_queue, valid_queue, num_classes = datasets.get_loaders(args, 'eval')

    # Set up the network.
    if os.path.isfile(args.genotype):
        logging.info('Loading genotype from: %s' % args.genotype)
        genotype = torch.load(args.genotype)
    else:
        logging.info('Loading genotype: %s' % args.genotype)
        genotype = eval('genotypes.%s' % args.genotype)
    if not isinstance(genotype, list):
        genotype = [genotype]
    model = Network(args.init_channels, num_classes, args.layers,
                    args.auxiliary, genotype, args.dataset)
    model = model.cuda()
    model = DDP(model, delay_allreduce=True)

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()
    logging.info('param size = %f M', utils.count_parameters_in_M(model))

    # Set up network weights optimizer.
    optimizer = torch.optim.SGD(
        model.parameters(),
        args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, float(args.epochs - args.warmup_epochs))

    # Train.
    global_step = 0
    for epoch in range(args.epochs):
        # Shuffle the sampler, update lrs.
        train_queue.sampler.set_epoch(epoch + args.seed)
        if epoch >= args.warmup_epochs:
            scheduler.step()

        # Training.
        model.module.drop_path_prob = args.drop_path_prob * epoch / args.epochs
        train_acc, train_obj, global_step = train(
            train_queue, model, criterion, optimizer, epoch, args.learning_rate,
            args.warmup_epochs, global_step)
        logging.info('train_acc %f', train_acc)
        writer.add_scalar('train/acc', train_acc, global_step)

        # Validation.
        valid_acc, valid_obj = infer(valid_queue, model, criterion)
        logging.info('valid_acc %f', valid_acc)
        writer.add_scalar('val/acc', valid_acc, global_step)
        writer.add_scalar('val/loss', valid_obj, global_step)

        if args.local_rank == 0:
            utils.save(model, os.path.join(args.save, 'weights.pt'))

    writer.flush()


def train(train_queue, model, criterion, optimizer, epoch, init_lr, warmup_epochs, global_step):
    """Update network weights on train set."""
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
        logits, logits_aux = model(data)
        loss = criterion(logits, target)
        if args.auxiliary:
            loss_aux = criterion(logits_aux, target)
            loss += args.auxiliary_weight * loss_aux

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
            writer.add_scalar('train/acc1', top1.avg, global_step)
            writer.add_scalar('train/lr', optimizer.state_dict()[
                              'param_groups'][0]['lr'], global_step)
        global_step += 1

    return top1.avg, objs.avg, global_step


def infer(valid_queue, model, criterion):
    """Run model in eval only mode."""
    objs = utils.AverageMeter()
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()

    model.eval()
    with torch.no_grad():
        for step, (data, target) in enumerate(valid_queue):
            data = data.cuda()
            target = target.cuda()

            logits, _ = model(data)
            loss = criterion(logits, target)

            prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
            n = data.size(0)
            objs.update(loss.item(), n)
            top1.update(prec1.item(), n)
            top5.update(prec5.item(), n)

            if step % args.report_freq == 0:
                logging.info('valid %03d %e %f %f', step,
                             objs.avg, top1.avg, top5.avg)

    return top1.avg, objs.avg


if __name__ == '__main__':
    main()
