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

import torch
import torch.nn as nn
from darts.operations import *
from darts.genotypes import PRIMITIVES
from util.utils import drop_path


class FactorizedCell(nn.Module):

    def __init__(self, genotype, C_prev_prev, C_prev, C, reduction, reduction_prev, concat):
        super(FactorizedCell, self).__init__()

        if reduction_prev:
            self.preprocess0 = FactorizedReduce(C_prev_prev, C, affine=True)
        else:
            self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0, affine=True)
        self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0, affine=True)

        self.structure = genotype
        self._compile(C, concat, reduction)

    def _compile(self, C, concat, reduction):
        self._steps = len(self.structure)
        self._concat = concat
        self.multiplier = len(concat)

        self._ops0 = nn.ModuleList()
        self._ops1 = nn.ModuleList()
        for step_structure in self.structure:
            ops_index = step_structure['ops']
            inputs_index = step_structure['inputs']

            stride = 2 if reduction and inputs_index[0] < 2 else 1
            op = OPS[PRIMITIVES[ops_index[0]]](C, stride, True)
            self._ops0 += [op]

            stride = 2 if reduction and inputs_index[1] < 2 else 1
            op = OPS[PRIMITIVES[ops_index[1]]](C, stride, True)
            self._ops1 += [op]

    def forward(self, s0, s1, drop_prob):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)

        states = [s0, s1]
        for i in range(self._steps):
            inputs_index = self.structure[i]['inputs']

            branch0 = self._ops0[i](states[inputs_index[0]])
            if self.training and drop_prob > 0 and not isinstance(self._ops0[i], Identity):
                branch0 = drop_path(branch0, drop_prob)

            branch1 = self._ops1[i](states[inputs_index[1]])
            if self.training and drop_prob > 0 and not isinstance(self._ops1[i], Identity):
                branch1 = drop_path(branch1, drop_prob)

            state = branch0 + branch1
            states += [state]

        return torch.cat([states[i] for i in self._concat], dim=1)


class AuxiliaryHeadCIFAR(nn.Module):

    def __init__(self, C, num_classes):
        """assuming input size 8x8"""
        super(AuxiliaryHeadCIFAR, self).__init__()
        self.features = nn.Sequential(
            nn.ReLU(inplace=True),
            # image size = 2 x 2
            nn.AvgPool2d(5, stride=3, padding=0, count_include_pad=False),
            nn.Conv2d(C, 128, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 768, 2, bias=False),
            nn.BatchNorm2d(768),
            nn.ReLU(inplace=True)
        )
        self.classifier = nn.Linear(768, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x.view(x.size(0), -1))
        return x


class AuxiliaryHeadImageNet(nn.Module):

    def __init__(self, C, num_classes, bn):
        """assuming input size 14x14"""
        super(AuxiliaryHeadImageNet, self).__init__()
        if bn:
            self.features = nn.Sequential(
                nn.ReLU(inplace=True),
                nn.AvgPool2d(5, stride=2, padding=0, count_include_pad=False),
                nn.Conv2d(C, 128, 1, bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 768, 2, bias=False),
                nn.BatchNorm2d(768),
                nn.ReLU(inplace=True)
            )
        else:
            self.features = nn.Sequential(
                nn.ReLU(inplace=True),
                nn.AvgPool2d(5, stride=2, padding=0, count_include_pad=False),
                nn.Conv2d(C, 128, 1, bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 768, 2, bias=False),
                # nn.BatchNorm2d(768),
                nn.ReLU(inplace=True)
            )
        self.classifier = nn.Linear(768, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x.view(x.size(0), -1))
        return x


class Network(nn.Module):

    def __init__(self, C, num_classes, layers, auxiliary, genotype, dataset):
        super(Network, self).__init__()
        self._layers = layers
        self._auxiliary = auxiliary
        stem_multiplier = 3

        if dataset in ['cifar10', 'cifar100']:
            C_curr = stem_multiplier*C
            self.stem = nn.Sequential(
                nn.Conv2d(3, C_curr, 3, padding=1, bias=False),
                nn.BatchNorm2d(C_curr)
            )
            C_prev_prev, C_prev, C_curr = C_curr, C_curr, C

        self.cells = nn.ModuleList()
        reduction_prev = False
        cell_type_idx = 0  # This increments after every reduction.

        for i in range(layers):
            if i in [layers//3, 2*layers//3]:
                C_curr *= 2
                if len(genotype) > 1:
                    cell_type_idx += 1

                reduction = True
                if isinstance(genotype[cell_type_idx].reduce[0], list):
                    genotype_layer = genotype[cell_type_idx].reduce[0]
                else:
                    genotype_layer = genotype[cell_type_idx].reduce
                concat = genotype[cell_type_idx].reduce_concat
            else:
                reduction = False
                if isinstance(genotype[cell_type_idx].normal[0], list):
                    genotype_layer = genotype[cell_type_idx].normal[0]
                else:
                    genotype_layer = genotype[cell_type_idx].normal
                concat = genotype[cell_type_idx].normal_concat

            cell = FactorizedCell(genotype_layer, C_prev_prev, C_prev,
                                  C_curr, reduction, reduction_prev, concat)
            reduction_prev = reduction
            self.cells += [cell]
            C_prev_prev, C_prev = C_prev, cell.multiplier*C_curr
            if i == 2*layers//3:
                C_to_auxiliary = C_prev

        if auxiliary:
            self.auxiliary_head = AuxiliaryHeadCIFAR(
                C_to_auxiliary, num_classes)
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_prev, num_classes)

    def forward(self, input):
        logits_aux = None
        s0 = s1 = self.stem(input)
        for i, cell in enumerate(self.cells):
            s0, s1 = s1, cell(s0, s1, self.drop_path_prob)
            if i == 2*self._layers//3:
                if self._auxiliary and self.training:
                    logits_aux = self.auxiliary_head(s1)
        out = self.global_pooling(s1)
        logits = self.classifier(out.view(out.size(0), -1))
        return logits, logits_aux


class NetworkImageNet(nn.Module):

    def __init__(self, C, num_classes, layers, auxiliary, genotype, bn=True):
        super(NetworkImageNet, self).__init__()
        self._layers = layers
        self._auxiliary = auxiliary

        self.stem0 = nn.Sequential(
            nn.Conv2d(3, C // 2, kernel_size=3,
                      stride=2, padding=1, bias=False),
            nn.BatchNorm2d(C // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(C // 2, C, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(C),
        )

        self.stem1 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(C, C, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(C),
        )

        C_prev_prev, C_prev, C_curr = C, C, C
        self.cells = nn.ModuleList()
        reduction_prev = True
        cell_type_idx = 0  # This increments after every reduction.

        # This is used for weights per_layer_cpunters
        cell_normal_idx, cell_reduce_idx = 0, 0
        for i in range(layers):
            if i in [layers//3, 2*layers//3]:
                C_curr *= 2
                if len(genotype) > 1:
                    cell_type_idx += 1

                reduction = True
                if isinstance(genotype[cell_type_idx].reduce[0], list):
                    genotype_layer = genotype[cell_type_idx].reduce[0]
                else:
                    genotype_layer = genotype[cell_type_idx].reduce
                concat = genotype[cell_type_idx].reduce_concat
            else:
                reduction = False
                if isinstance(genotype[cell_type_idx].normal[0], list):
                    genotype_layer = genotype[cell_type_idx].normal[0]
                else:
                    genotype_layer = genotype[cell_type_idx].normal
                concat = genotype[cell_type_idx].normal_concat

            cell = FactorizedCell(genotype_layer, C_prev_prev, C_prev,
                                  C_curr, reduction, reduction_prev, concat)
            reduction_prev = reduction
            self.cells += [cell]
            C_prev_prev, C_prev = C_prev, cell.multiplier * C_curr
            if i == 2 * layers // 3:
                C_to_auxiliary = C_prev

        if auxiliary:
            self.auxiliary_head = AuxiliaryHeadImageNet(
                C_to_auxiliary, num_classes, bn)
        self.global_pooling = nn.AvgPool2d(7)
        self.classifier = nn.Linear(C_prev, num_classes)

    def forward(self, input):
        logits_aux = None
        s0 = self.stem0(input)
        s1 = self.stem1(s0)
        for i, cell in enumerate(self.cells):
            s0, s1 = s1, cell(s0, s1, self.drop_path_prob)
            if i == 2 * self._layers // 3:
                if self._auxiliary and self.training:
                    logits_aux = self.auxiliary_head(s1)
        out = self.global_pooling(s1)
        logits = self.classifier(out.view(out.size(0), -1))
        return logits, logits_aux
