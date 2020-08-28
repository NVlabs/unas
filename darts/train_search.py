# -----------------------------------------------------------------
# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# This file has been modified from a file with the same name in the
# DARTS library (licensed under the Apache License, Version 2.0):
#
# https://github.com/quark0/darts
#
# The Apache License for the original version of this file can be
# found in this directory. The modifications to this file are subject
# to the same Apache License, Version 2.0.
# -----------------------------------------------------------------


"""Search for the best cell."""
import argparse
import os
import sys
sys.path.append('../')
from apex.parallel import DistributedDataParallel as DDP
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.utils
from unas import UNAS


parser = argparse.ArgumentParser('Cell search')
# Dataset choices.
parser.add_argument('--data', type=str, default='/tmp/data/',
                    help='location of the data corpus')
parser.add_argument('--dataset', type=str, default='cifar10',
                    choices=['cifar10', 'cifar100', 'imagenet'],
                    help='which dataset to use')
parser.add_argument('--meta_loss', type=str, default='default',
                    choices=['rebar', 'reinforce', 'relax'],
                    help='which meta loss to use')
parser.add_argument('--train_portion', type=float, default=0.9,
                    help='portion of training data')
parser.add_argument('--val_arch_update', action='store_true', default=False,
                    help='if True, architecture is updated on validations batches')
parser.add_argument('--num_cell_types', type=int, default=1,
                    choices=[1, 3],
                    help='how many types of cells')
parser.add_argument('--num_ops', type=int, default=7,
                    choices=[3, 4, 5, 7, 8],
                    help='how many types of operations')
# Optimization choices.
parser.add_argument('--batch_size', type=int, default=32,
                    help='batch size')
parser.add_argument('--learning_rate', type=float, default=3e-4,
                    help='init learning rate')
parser.add_argument('--learning_rate_min', type=float, default=1e-4,
                    help='min learning rate')
parser.add_argument('--weight_decay', type=float, default=3e-4,
                    help='weight decay')
parser.add_argument('--arch_learning_rate', type=float, default=1e-3,               # org: 3e-3
                    help='learning rate for arch encoding')
parser.add_argument('--arch_learning_rate_min', type=float, default=3e-4,           # org: 3e-4
                    help='learning rate for arch encoding')
parser.add_argument('--arch_weight_decay', type=float, default=1e-6,
                    help='weight decay for arch encoding')
parser.add_argument('--epochs', type=int, default=85,
                    help='num of training epochs')
parser.add_argument('--init_epochs', type=int, default=15,
                    help='num of initial training epochs')
parser.add_argument('--grad_clip', type=float, default=5,
                    help='gradient clipping')
parser.add_argument('--cutout', action='store_true', default=False,
                    help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16,
                    help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.3,
                    help='drop path probability')
parser.add_argument('--scale_lr', action='store_true', default=False,
                    help='scale learning rate with num gpus?')
# Alpha loss
parser.add_argument('--alpha_loss', action='store_true', default=True,
                    help='add weights loss on alphas')
parser.add_argument('--alpha_loss_iter', type=float, default=5000,
                    help='number of iterations that entropy coeff will be annealed')
parser.add_argument('--alpha_loss_lambda', type=float, default=0.2,
                    help='multiplier on weights loss')
# Generalization Error
parser.add_argument('--gen_error_alpha', action='store_true', default=False,
                    help='add generalization error to loss when updating weights')
parser.add_argument('--gen_error_alpha_lambda', type=float, default=0.5,
                    help='coefficient to combine train/val loss with generalization error loss')
# Gumbel options.
parser.add_argument('--same_alpha_minibatch', action='store_true', default=False,
                    help='if True, uses same weights sample for every sample in minibatch')
parser.add_argument('--gumbel_soft_temp', type=float, default=0.4,
                    help='temperature for sampling with gumbel-softmax')
parser.add_argument('--gsm_type', type=str, default='original',
                    choices=['original', 'rebar'],
                    help='which GSM sampling to use')
parser.add_argument('--gsm_soften_eps', type=float, default=0.0,
                    help='epsilon used for softening GSM probablities')
# Architecture choices.
parser.add_argument('--init_channels', type=int, default=16,
                    help='num of init channels')
parser.add_argument('--layers', type=int, default=8,
                    help='total number of layers')
parser.add_argument('--steps', type=int, default=4,
                    help='total number of steps in each layer')
# Latency loss
parser.add_argument('--target_latency', type=float, default=0.,
                    help='To enable optimizing latency, set the target latency to a '
                    'value greater than 0. (in milliseconds)')
parser.add_argument('--latency_iter', type=float, default=30000,
                    help='number of iterations for warming up latency')
parser.add_argument('--latency_coeff', type=float, default=0.1,
                    help='The coefficient used for the latency loss')
# Logging choices.
parser.add_argument('--report_freq', type=float, default=100,
                    help='report frequency')
parser.add_argument('--model_path', type=str, default='saved_models',
                    help='path to save the model')
parser.add_argument('--save', type=str, default='EXP1',
                    help='experiment name')
parser.add_argument('--root_dir', type=str, default='/tmp/checkpoints/',
                    help='root directory')
# Misc.
parser.add_argument('--seed', type=int, default=2,
                    help='random seed')
parser.add_argument('--gpu', type=int, default=0,
                    help='gpu device id')
# DDP.
parser.add_argument('--local_rank', type=int, default=0,
                    help='rank of process')

args = parser.parse_args()
assert args.meta_loss

# Set up primitives.
from darts.genotypes import set_primitives
set_primitives(args.num_ops)
from util import utils
from util import datasets
from darts.model_search import Network
from darts.alphas import Alpha
from darts.genotypes import PRIMITIVES


# Fix some input params.
args.multiplier = args.steps
if args.meta_loss == 'rebar' or args.meta_loss == 'relax':
    args.gsm_type = 'rebar'


# Set up DDP.
args.distributed = True
torch.cuda.set_device(args.local_rank)
torch.distributed.init_process_group(backend='nccl', init_method='env://')
args.world_size = torch.distributed.get_world_size()

# Set up logging.
assert args.root_dir
args.save = args.root_dir + '/search-{}'.format(args.save)
if args.local_rank == 0:
    utils.create_exp_dir(args.save)
logging = utils.Logger(args.local_rank, args.save)
writer = utils.Writer(args.local_rank, args.save)


def main():
    """Do everything!"""
    # Scale learning rate based on global batch size.
    if args.scale_lr:
        scale = float(args.batch_size * args.world_size) / 64.0
        args.learning_rate = scale * args.learning_rate
        args.arch_learning_rate = scale * args.arch_learning_rate

    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)

    np.random.seed(args.seed)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled = True
    torch.cuda.manual_seed(args.seed)
    logging.info('gpu device = %d', args.gpu)
    logging.info('args = %s', args)

    # Get data loaders.
    train_queue, valid_queue, num_classes = datasets.get_loaders(args, 'search')

    # Set up the network and criterion.
    model = Network(args.init_channels, num_classes, args.layers,
                    num_cell_types=args.num_cell_types,
                    dataset=args.dataset,
                    steps=args.steps,
                    multiplier=args.multiplier)
    model = model.cuda()
    alpha = Alpha(num_normal=1,
                  num_reduce=1,
                  num_op=len(PRIMITIVES),
                  num_nodes=args.steps,
                  gsm_soften_eps=args.gsm_soften_eps,
                  gsm_temperature=args.gumbel_soft_temp,
                  gsm_type=args.gsm_type,
                  same_alpha_minibatch=args.same_alpha_minibatch)
    alpha = alpha.cuda()
    model = DDP(model, delay_allreduce=True)
    alpha = DDP(alpha, delay_allreduce=True)

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()
    logging.info('param size = %fM ', utils.count_parameters_in_M(model))
    writer.add_scalar('temperature', args.gumbel_soft_temp)

    # Get weight params, and arch params.
    weight_params = [p for p in model.parameters()]
    arch_params = [p for p in alpha.parameters()]

    logging.info('#Weight params: %d, #Arch params: %d' %
                 (len(weight_params), len(arch_params)))

    # Initial weight pretraining.
    def run_train_init():
        logging.info('running init epochs.')
        opt = torch.optim.Adam(
            weight_params,
            args.learning_rate,
            weight_decay=args.weight_decay)
        for e in range(args.init_epochs):
            # Shuffle the sampler.
            train_queue.sampler.set_epoch(e + args.seed)
            train_acc, train_obj = train_init(
                train_queue, model, alpha, criterion, opt, weight_params)
            logging.info('train_init_acc %f', train_acc)
            valid_acc, valid_obj = infer(valid_queue, model, alpha, criterion)
            logging.info('valid_init_acc %f', valid_acc)
            memory_cached, memory_alloc = utils.get_memory_usage(device=0)
            logging.info('memory_cached %0.3f memory_alloc %0.3f' % (memory_cached, memory_alloc))

    if args.init_epochs:
        run_train_init()

    # Set up network weights optimizer.
    optimizer = torch.optim.Adam(
        weight_params, args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, float(args.epochs), eta_min=args.learning_rate_min)

    # Wrap model in architecture learner.
    nas = UNAS(model, alpha, args, writer, logging)

    global_step = 0
    for epoch in range(args.epochs):
        # Shuffle the sampler, update lrs.
        train_queue.sampler.set_epoch(epoch + args.seed)
        scheduler.step()
        nas.arch_scheduler.step()

        # Logging.
        if args.local_rank == 0:
            memory_cached, memory_alloc = utils.get_memory_usage(device=0)
            writer.add_scalar('memory/cached', memory_cached, global_step)
            writer.add_scalar('memory/alloc', memory_alloc, global_step)
            logging.info('memory_cached %0.3f memory_alloc %0.3f' % (memory_cached, memory_alloc))
            logging.info('epoch %d lr %e', epoch, scheduler.get_lr()[0])
            writer.add_scalar('train/lr', scheduler.get_lr()[0], global_step)
            writer.add_scalar('train/arc_lr', nas.arch_scheduler.get_lr()[0], global_step)

            genotypes = alpha.module.genotype()
            logging.info('genotype:')
            for genotype in genotypes:
                logging.info('normal:')
                logging.info(genotype.normal)
                logging.info('reduce:')
                logging.info(genotype.reduce)

            # alphas_normal, alphas_reduce = weights.module.add_based_alpha_paired_input()
            for l in range(len(alpha.module.alphas_normal)):
                fig = alpha.module.plot_alphas(alpha.module.alphas_normal[l])
                writer.add_figure('weights/disp_normal_%d' % l, fig, global_step)
            for l in range(len(alpha.module.alphas_reduce)):
                fig = alpha.module.plot_alphas(alpha.module.alphas_reduce[l])
                writer.add_figure('weights/disp_reduce_%d' % l, fig, global_step)

        # Training.
        train_acc, train_obj, global_step = train(
            train_queue, valid_queue, model, alpha, nas, criterion,
            optimizer, global_step, weight_params, args.seed)
        logging.info('train_acc %f', train_acc)
        writer.add_scalar('train/acc', train_acc, global_step)

        # Validation.
        valid_queue.sampler.set_epoch(0)
        valid_acc, valid_obj = infer(valid_queue, model, alpha, criterion)
        logging.info('valid_acc %f', valid_acc)
        writer.add_scalar('val/acc', valid_acc, global_step)
        writer.add_scalar('val/loss', valid_obj, global_step)

        if args.local_rank == 0:
            utils.save(model, os.path.join(args.save, 'weights.pt'))
            torch.save(alpha.module.genotype(), os.path.join(args.save, 'genotype.pt'))

    writer.flush()


def train(train_queue, valid_queue, model, alpha, nas, criterion, optimizer, global_step, weight_params, seed):
    """Update network weights on train set and architecture on val set."""
    objs = utils.AverageMeter()
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()
    nas.reset_counter()

    # Init meta queue iterator.
    valid_queue.sampler.set_epoch(global_step + seed)
    valid_queue_iterator = iter(valid_queue)

    for step, (data, target) in enumerate(train_queue):
        model.train()
        n = data.size(0)

        # Use the same minibatch for updating weights as well as arch params.
        data = data.cuda()
        target = target.cuda()

        # update architecture
        weights = alpha(data.size(0))
        weights_no_grad = alpha.module.clone_weights(weights)
        # Get a random minibatch from the valid queue with replacement.
        # Use this to update the architecture.
        if args.val_arch_update or args.gen_error_alpha:
            try:
                input_valid, target_valid = next(valid_queue_iterator)
            except:
                valid_queue.sampler.set_epoch(global_step + seed)
                valid_queue_iterator = iter(valid_queue)
                input_valid, target_valid = next(valid_queue_iterator)

            input_valid = input_valid.cuda()
            target_valid = target_valid.cuda()
            if args.gen_error_alpha:
                nas.step(data, target, global_step, weights,
                         input_valid, target_valid, optimizer)
            else:
                nas.step(input_valid, target_valid,
                         global_step, weights)
        else:
            nas.step(data, target, global_step, weights)

        optimizer.zero_grad()
        logits = model(data, weights_no_grad)
        loss = criterion(logits, target)
        dummy = sum([torch.sum(param) for param in model.parameters()])
        loss += dummy * 0.
        loss.backward()
        nn.utils.clip_grad_norm_(weight_params, args.grad_clip)
        optimizer.step()

        # Calculate the accuracy.
        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        objs.update(loss.data, n)
        top1.update(prec1.data, n)
        top5.update(prec5.data, n)

        if (step + 1) % args.report_freq == 0 or step == len(train_queue) - 1:
            objs_avg = utils.reduce_tensor(objs.avg, args.world_size)
            top1_avg = utils.reduce_tensor(top1.avg, args.world_size)
            top5_avg = utils.reduce_tensor(top5.avg, args.world_size)

            logging.info('train %03d %e %f %f', step,
                         objs_avg, top1_avg, top5_avg)
            writer.add_scalar('train/loss', objs_avg, global_step)
            writer.add_scalar('train/acc1', top1_avg, global_step)
            writer.add_scalar('train/lr', optimizer.state_dict()[
                              'param_groups'][0]['lr'], global_step)
        global_step += 1

    return top1_avg, objs_avg, global_step


def infer(valid_queue, model, alpha, criterion):
    """Run model in eval only mode."""
    objs = utils.AverageMeter()
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()
    # model.eval()

    with torch.no_grad():
        for step, (data, target) in enumerate(valid_queue):
            n = data.size(0)
            data = data.cuda()
            target = target.cuda()

            weights = alpha(data.size(0))
            logits = model(data, weights)
            loss = criterion(logits, target)

            # Calculate the accuracy.
            prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))

            objs.update(loss.data, n)
            top1.update(prec1.data, n)
            top5.update(prec5.data, n)

            if step % args.report_freq == 0 or step == len(valid_queue) - 1:
                objs_avg = utils.reduce_tensor(objs.avg, args.world_size)
                top1_avg = utils.reduce_tensor(top1.avg, args.world_size)
                top5_avg = utils.reduce_tensor(top5.avg, args.world_size)
                logging.info('valid %03d %e %f %f', step,
                             objs_avg, top1_avg, top5_avg)

    return top1_avg, objs_avg


def train_init(train_queue, model, alpha, criterion, optimizer, weight_params):
    """Update network weights on train set and architecture on val set."""
    objs = utils.AverageMeter()
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()

    for step, (data, target) in enumerate(train_queue):
        model.train()
        n = data.size(0)

        # Update network weights using the train set.
        data = data.cuda()
        target = target.cuda()

        weights = alpha(data.size(0))
        weights_no_grad = alpha.module.clone_weights(weights)

        optimizer.zero_grad()
        logits = model(data, weights_no_grad)
        loss = criterion(logits, target)
        dummy = sum([torch.sum(param) for param in model.parameters()])
        loss += dummy * 0.
        loss.backward()
        nn.utils.clip_grad_norm_(weight_params, args.grad_clip)
        optimizer.step()

        # Calculate the accuracy.
        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))

        objs.update(loss.data, n)
        top1.update(prec1.data, n)
        top5.update(prec5.data, n)

        if (step + 1) % args.report_freq == 0 or step == len(train_queue) - 1:
            objs_avg = utils.reduce_tensor(objs.avg, args.world_size)
            top1_avg = utils.reduce_tensor(top1.avg, args.world_size)
            top5_avg = utils.reduce_tensor(top5.avg, args.world_size)
            logging.info('train_init %03d %e %f %f', step,
                         objs_avg, top1_avg, top5_avg)

    return top1_avg, objs_avg


if __name__ == '__main__':
    main()