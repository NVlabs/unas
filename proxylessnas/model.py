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

import torch
import torch.nn as nn
from proxylessnas.operations import *


class Network(nn.Module):
    def __init__(self, genotype, num_classes=1000, layers=21, width_multiplier=1.0, use_proxylessnas_c=True):
        super(Network, self).__init__()
        self._layers = layers

        # set use_proxylessnas_c to True when building proxylessnas model.
        if use_proxylessnas_c:
            C = [40, 24, 32, 56, 112, 128, 256, 432, 1728]
        else:
            C = [24, 12, 18, 24, 48, 72, 120, 240, 960]
        C = [int(round(c * width_multiplier)) for c in C]
        self.channel_change_layers = [1, 5, 9, 13, 17, 21]
        self.stride_2_layers = [1, 5, 9, 17]
        # Use the Proxyless stem.
        self.stem = nn.Sequential(
            ConvLayer(3, C[0], kernel_size=3, stride=2, dilation=1,
                      groups=1, bias=False, has_shuffle=False,
                      use_bn=True, act_func='relu6', dropout_rate=0,
                      ops_order='weight_bn_act'),
            MobileInvertedResidualBlock(
                C[0], C[1], kernel_size=3, stride=1, expand_ratio=1)
        )
        C_in = C[1]
        C_idx = 1

        # Create the blocks.
        self.cells = nn.ModuleList()
        normal_idx, reduce_idx = -1, -1
        for i in range(1, layers + 1):
            if i in self.channel_change_layers:
                reduce_idx += 1
                stride = 2 if i in self.stride_2_layers else 1
                # Update the channel count.
                C_idx += 1
                C_out = C[C_idx]

                cell_type = genotype.reduce[reduce_idx]
                cell = OPS[cell_type](C_in, C_out, stride)
            else:
                normal_idx += 1
                stride = 1

                cell_type = genotype.normal[normal_idx]
                cell = OPS[cell_type](C_in, C_out, stride)

            self.cells += [cell]
            C_in = C_out

        # Create the final output layer.
        self.feature_mix_layer = ConvLayer(
            C[-2], C[-1], kernel_size=1, stride=1, dilation=1,
            groups=1, bias=False, has_shuffle=False,
            use_bn=True, act_func='relu6', dropout_rate=0,
            ops_order='weight_bn_act')
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = LinearLayer(
            C[-1], num_classes, bias=True, use_bn=False,
            act_func=None, dropout_rate=0, ops_order='weight_bn_act')

    def forward(self, data):
        x = self.stem(data)
        for i, cell in enumerate(self.cells):
            x = cell(x)
        x = self.feature_mix_layer(x)
        x = self.global_pooling(x)
        x = x.view(x.size(0), -1)
        logits = self.classifier(x)
        return logits
