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

import numpy as np
from proxylessnas.operations import *
from proxylessnas.genotypes import BLOCK_PRIMITIVES, REDUCTION_PRIMITIVES


class MixedOp(nn.Module):
    def __init__(self, Cin, Cout, stride, primitives, ops=OPS, reduction=False, dropout_prob=0.1):
        super(MixedOp, self).__init__()
        self._ops = nn.ModuleList()
        self.reduction = reduction
        for primitive in primitives:
            op = ops[primitive](Cin, Cout, stride)
            if primitive == 'mic_zero':  # Following P-DARTS add noise to skip operation.
                op = nn.Sequential(op, nn.Dropout(dropout_prob))
            self._ops.append(op)

    def forward(self, x, weights):
        # discrete case, single weight
        total_sum = 0
        for idx in range(len(self._ops)):
            op = self._ops[idx]
            w = weights[:, idx]
            # remove computation for very small w's, gradient of GSM is already small for small w.
            if (w.size(0) == 1 and float(w[0]) < 5e-2) or torch.max(w) < 5e-2:
                total_sum += 0.
            else:
                y = op(x)
                w = w.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
                total_sum += (w * y)
        return total_sum

    def fast_forward(self, x, idx):
        """ This function is the similar to forward, with the main difference that it only calls the operation
        indicated by idx instead of all the operations. """
        return self._ops[idx](x)


class Network(nn.Module):
    def __init__(self, num_classes=1000, layers=21, width_multiplier=1.,
                 dataset='imagenet', same_alpha_minibatch=False,
                 gumbel_soft_temp=0.4, gsm_type='original', gsm_soften_eps=0.,
                 use_proxylessnas_c=True):
        super(Network, self).__init__()
        self._num_classes = num_classes
        self.layers = layers
        self._dataset = dataset
        self._temperature = gumbel_soft_temp
        self._gsm_soften_eps = gsm_soften_eps
        self._same_alpha_minibatch = same_alpha_minibatch
        self._alpha_per_layer = True
        self.gsm_type = gsm_type

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
        for i in range(1, layers + 1):
            reduction = False
            primitives = BLOCK_PRIMITIVES
            stride = 1

            if i in self.channel_change_layers:
                reduction = True
                primitives = REDUCTION_PRIMITIVES
                stride = 2 if i in self.stride_2_layers else 1
                # Update the channel count.
                C_idx += 1
                C_out = C[C_idx]

            cell = MixedOp(C_in, C_out, stride, primitives, reduction=reduction)
            self.cells += [cell]
            C_in = C_out

        # Create the final output layer.
        self.feature_mix_layer = ConvLayer(
            C[-2], C[-1], kernel_size=1, stride=1, dilation=1,
            groups=1, bias=False, has_shuffle=False,
            use_bn=True, act_func='relu6', dropout_rate=0,
            ops_order='weight_bn_act')
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = LinearLayer(C[-1], num_classes, bias=True, use_bn=False,
                                      act_func=None, dropout_rate=0, ops_order='weight_bn_act')

    def forward(self, data, all_weights):
        s = self.stem(data)

        cell_normal_idx, cell_reduce_idx = 0, 0
        for c, cell in enumerate(self.cells):
            if cell.reduction:
                weights = all_weights['reduce'][cell_reduce_idx]
                cell_reduce_idx += 1
            else:
                weights = all_weights['normal'][cell_normal_idx]
                cell_normal_idx += 1

            s = cell(s, weights)

        out = self.feature_mix_layer(s)
        out = self.global_pooling(out)
        out = out.view(out.size(0), -1)
        logits = self.classifier(out)
        return logits

    def fast_forward(self, data, indices):
        """ This function is the similar to forward, with the main difference that it only calls ops in each cell
        indicated by idx instead of all the operations. """
        s = self.stem(data)
        for cell, idx in zip(self.cells, indices):
            s = cell.fast_forward(s, idx)

        out = self.feature_mix_layer(s)
        out = self.global_pooling(out)
        out = out.view(out.size(0), -1)
        logits = self.classifier(out)
        return logits

    def get_indices(self, discrete_weights):
        cell_normal_idx, cell_reduce_idx = 0, 0
        weight_indices = []
        for c, cell in enumerate(self.cells):
            if cell.reduction:
                weights = discrete_weights['reduce'][cell_reduce_idx]
                cell_reduce_idx += 1
            else:
                weights = discrete_weights['normal'][cell_normal_idx]
                cell_normal_idx += 1

            assert weights.size(0) == 1 and float(torch.max(weights[0, :])) == 1., 'weights should be discrete'
            idx = torch.argmax(weights[0])
            idx = int(idx)
            weight_indices.append(idx)

        return weight_indices
