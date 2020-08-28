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

import logging
import os
import shutil
import time
import sys
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from tensorboardX import SummaryWriter


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt


class ExpMovingAvgrageMeter(object):
    def __init__(self, momentum=0.9):
        self.momentum = momentum
        self.reset()

    def reset(self):
        self.avg = 0

    def update(self, val):
        self.avg = (1. - self.momentum) * self.avg + self.momentum * val


def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0/batch_size))
    return res


class Cutout(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img


def count_parameters_in_M(model):
    return np.sum(np.prod(v.size()) for name, v in model.named_parameters() if "auxiliary" not in name)/1e6


def save_checkpoint(state, is_best, save):
    filename = os.path.join(save, 'checkpoint.pth.tar')
    torch.save(state, filename)
    if is_best:
        best_filename = os.path.join(save, 'model_best.pth.tar')
        shutil.copyfile(filename, best_filename)


def save(model, model_path):
    torch.save(model.state_dict(), model_path)


def load(model, model_path):
    model.load_state_dict(torch.load(model_path))


def drop_path(x, drop_prob):
    if drop_prob > 0.:
        keep_prob = 1.-drop_prob
        mask = torch.cuda.FloatTensor(
            x.size(0), 1, 1, 1).bernoulli_(keep_prob)
        x = x / keep_prob
        x = x * mask
    return x


def create_exp_dir(path, scripts_to_save=None):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    print('Experiment dir : {}'.format(path))

    if scripts_to_save is not None:
        if not os.path.exists(os.path.join(path, 'scripts')):
            os.mkdir(os.path.join(path, 'scripts'))
        for script in scripts_to_save:
            dst_file = os.path.join(path, 'scripts', os.path.basename(script))
            shutil.copyfile(script, dst_file)


class ClassErrorMeter(object):
    def __init__(self):
        super(ClassErrorMeter, self).__init__()
        self.class_counter = {}

    def add(self, output, target):
        _, pred = output.max(dim=1)

        target = list(target.cpu().numpy())
        pred = list(pred.cpu().numpy())

        for t, p in zip(target, pred):
            if t not in self.class_counter:
                self.class_counter[t] = {'num': 0, 'correct': 0}
            self.class_counter[t]['num'] += 1
            if t == p:
                self.class_counter[t]['correct'] += 1

    def value(self, method):
        print('Error type: ', method)
        if method == 'per_class':
            mean_accuracy = 0
            for t in self.class_counter:
                class_accuracy = float(self.class_counter[t]['correct']) / \
                    self.class_counter[t]['num']
                mean_accuracy += class_accuracy
            mean_accuracy /= len(self.class_counter)
            output = mean_accuracy * 100
        elif method == 'overall':
            num_total, num_correct = 0, 0
            for t in self.class_counter:
                num_total += self.class_counter[t]['num']
                num_correct += self.class_counter[t]['correct']
            output = float(num_correct) / num_total * 100
        return [100 - output]


class Logger(object):
    def __init__(self, rank, save):
        self.rank = rank
        if self.rank == 0:
            log_format = '%(asctime)s %(message)s'
            logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                                format=log_format, datefmt='%m/%d %I:%M:%S %p')
            fh = logging.FileHandler(os.path.join(save, 'log.txt'))
            fh.setFormatter(logging.Formatter(log_format))
            logging.getLogger().addHandler(fh)
            self.start_time = time.time()

    def info(self, string, *args):
        if self.rank == 0:
            elapsed_time = time.time() - self.start_time
            elapsed_time = time.strftime(
                '(Elapsed: %H:%M:%S) ', time.gmtime(elapsed_time))
            if isinstance(string, str):
                string = elapsed_time + string
            else:
                logging.info(elapsed_time)
            logging.info(string, *args)


class Writer(object):
    def __init__(self, rank, save):
        self.rank = rank
        if self.rank == 0:
            try:
                self.writer = SummaryWriter(log_dir=save, flush_secs=20)
            except:
                self.writer = SummaryWriter(logdir=save, flush_secs=20)

    def add_scalar(self, *args, **kwargs):
        if self.rank == 0:
            self.writer.add_scalar(*args, **kwargs)

    def add_figure(self, *args, **kwargs):
        if self.rank == 0:
            self.writer.add_figure(*args, **kwargs)

    def flush(self):
        if self.rank == 0:
            self.writer.flush()


def reduce_tensor(tensor, world_size):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= world_size
    return rt


def get_memory_usage(device=None):
    try:
        memory_cached = torch.cuda.max_memory_cached(device) * 1e-9
        memory_alloc = torch.cuda.max_memory_allocated(device) * 1e-9
        torch.cuda.reset_max_memory_allocated(device)
        torch.cuda.reset_max_memory_cached(device)
    except Exception:
        memory_cached, memory_alloc = 0., 0.
    return memory_cached, memory_alloc


if __name__ == '__main__':
    avg_meter = ExpMovingAvgrageMeter(momentum=0.9)
    for i in range(100):
        avg_meter.update(i)
        print(avg_meter.avg)
