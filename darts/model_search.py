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


import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from darts.operations import *
from darts.genotypes import PRIMITIVES


class MixedOp(nn.Module):
    def __init__(self, C, stride, primitives=PRIMITIVES, ops=OPS):
        super(MixedOp, self).__init__()
        self._ops = nn.ModuleList()
        for primitive in primitives:
            op = ops[primitive](C, stride, False)
            if 'pool' in primitive and False:
                op = nn.Sequential(op, nn.BatchNorm2d(C, affine=False))
            self._ops.append(op)

    def forward(self, x, weights):
        total_sum = 0.
        for idx in range(len(self._ops)):
            op = self._ops[idx]
            w = weights[:, idx]

            # remove computation very small w's when
            if (w.size(0) == 1 and float(w[0]) < 5e-2) or torch.max(w) < 5e-2:
                total_sum += 0.
            else:
                y = op(x)
                w = w.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
                total_sum += (w * y)

        return total_sum


class Cell(nn.Module):
    def __init__(self, steps, multiplier, C_prev_prev, C_prev, C, reduction, reduction_prev):
        super(Cell, self).__init__()

        def get_pre_process(CPP, CP):
            if reduction_prev:
                preprocess0 = FactorizedReduce(CPP, C, affine=False)
            else:
                preprocess0 = ReLUConvBN(CPP, C, 1, 1, 0, affine=False)

            preprocess1 = ReLUConvBN(CP, C, 1, 1, 0, affine=False)

            return preprocess0, preprocess1

        self.reduction = reduction
        self.preprocess0, self.preprocess1 = get_pre_process(C_prev_prev, C_prev)

        self._steps = steps
        self._multiplier = multiplier
        self._num_edges = sum(1 for i in range(steps) for n in range(2 + i))

        self._ops = nn.ModuleList()
        self._bns = nn.ModuleList()
        for i in range(self._steps):
            for j in range(2+i):
                stride = 2 if reduction and j < 2 else 1
                op = MixedOp(C, stride)
                self._ops.append(op)

    def forward(self, s0, s1, weights, params, layer_idx):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)

        states = [s0, s1]
        offset = 0
        for i in range(self._steps):
            s = 0
            for j, h in enumerate(states):
                s += self._ops[offset + j](h, weights[offset + j])
            offset += len(states)
            states.append(s)

        return torch.cat(states[-self._multiplier:], dim=1)


class CellPairedInput(Cell):
    def __init__(self, steps, multiplier, C_prev_prev, C_prev, C, reduction, reduction_prev):
        super(CellPairedInput, self).__init__(
            steps, multiplier, C_prev_prev, C_prev, C, reduction, reduction_prev)

    def forward(self, s0, s1, weights):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)
        num_steps = self._steps
        multiplier = self._multiplier

        states = [s0, s1]
        offset = 0
        for i in range(num_steps):
            s = 0
            for j, h in enumerate(states):
                ww = weights[i]['input_op0'][:, j] + weights[i]['input_op1'][:, j]
                s += self._ops[offset + j](h, ww)

            offset += len(states)
            states.append(s)

        return torch.cat(states[-multiplier:], dim=1)


class Network(nn.Module):
    def __init__(self, C, num_classes, layers, num_cell_types=1, steps=3,
                 multiplier=3, stem_multiplier=3, dataset='cifar10'):
        super(Network, self).__init__()
        self._C = C
        self._num_classes = num_classes
        self._layers = layers
        self._steps = steps
        self._multiplier = multiplier
        self._dataset = dataset
        self._num_cell_types = num_cell_types
        self._num_edges = sum(1 for i in range(steps) for n in range(2 + i))

        if dataset in ['cifar10', 'cifar100']:
            C_curr = stem_multiplier*C
            self.stem = nn.Sequential(
                nn.Conv2d(3, C_curr, 3, padding=1, bias=False),
                nn.BatchNorm2d(C_curr)
            )
            reduction_prev = False
            C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
        elif dataset in ['tim20', 'tim200']:
            C_curr = stem_multiplier*C
            self.stem = nn.Sequential(
                nn.Conv2d(3, C_curr // 2, kernel_size=3,
                          stride=2, padding=1, bias=False),
                nn.BatchNorm2d(C_curr // 2),
                nn.ReLU(inplace=True),
                nn.Conv2d(C_curr // 2, C_curr, kernel_size=3,
                          stride=1, padding=1, bias=False),
                nn.BatchNorm2d(C_curr),
            )
            reduction_prev = False
            C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
        elif dataset == 'imagenet':
            # Input image is of size 224 x 224.
            C_curr = stem_multiplier * C
            self.stem0 = nn.Sequential(
                nn.Conv2d(3, C_curr // 2, kernel_size=3,
                        stride=2, padding=1, bias=False),
                nn.BatchNorm2d(C_curr // 2),
                nn.ReLU(inplace=True),
                nn.Conv2d(C_curr // 2, C_curr, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(C_curr),
            )
            self.stem1 = nn.Sequential(
                nn.ReLU(inplace=True),
                nn.Conv2d(C_curr, C_curr, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(C_curr),
            )
            reduction_prev = True
            C_prev_prev, C_prev, C_curr = C_curr, C_curr, C_curr

        self.cells = nn.ModuleList()
        for i in range(layers):
            if i in [layers//3, 2*layers//3]:
                C_curr *= 2
                reduction = True
            else:
                reduction = False

            cell = CellPairedInput(steps, multiplier, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)

            reduction_prev = reduction
            self.cells += [cell]
            C_prev_prev, C_prev = C_prev, multiplier*C_curr

        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_prev, num_classes)

    def forward(self, data, all_weights, return_feats=False):
        if self._dataset == 'imagenet':
            s0 = self.stem0(data)
            s1 = self.stem1(s0)
        else:
            s0 = s1 = self.stem(data)
        
        # This is used for weights per_layer_cpunters
        cell_normal_idx, cell_reduce_idx = 0, 0
        self.flops = 0
        for cell in self.cells:
            if cell.reduction:
                weights = all_weights['reduce'][cell_reduce_idx]
            else:
                weights = all_weights['normal'][cell_normal_idx]

            s0, s1 = s1, cell(s0, s1, weights)

        if isinstance(s1, list):
            s1 = torch.cat(s1, dim=1)
        out = self.global_pooling(s1)
        out = out.view(out.size(0), -1)
        if return_feats:
            return out

        logits = self.classifier(out)
        return logits

