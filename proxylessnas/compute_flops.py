# ---------------------------------------------------------------
# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for UNAS. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

import torch
import numpy as np
from ptflops import get_model_complexity_info

from proxylessnas import genotypes
from proxylessnas.model import Network


def find_flops(genotype, layers, width_multiplier):
    # Set up the network.
    model = Network(genotype, num_classes=1000, layers=layers, width_multiplier=width_multiplier)
    model.drop_path_prob = 0

    with torch.cuda.device(0):
        flops, params = get_model_complexity_info(
            model, (3, 224, 224), as_strings=False, print_per_layer_stat=False)
        print('width mult: %f, Flops: %.2f, Params: %.2f' %
              (width_multiplier, flops/1e6, params/1e6))
        return flops, params


def find_max_channels(genotype, layers, max_flops):
    best_flops, best_params, best_C = None, None, None
    print(max_flops // 1e6)
    if max_flops // 1e6 == 600:
        start, end = 0.5, 5
    elif max_flops // 1e6 == 4000:
        start, end = 100, 150
    for width in np.arange(start, end, 0.02):
        flops, params = find_flops(genotype, layers, width)
        if flops < max_flops:
            best_width = width
            best_flops = flops
            best_params = params
        else:
            break
    return best_flops, best_params, best_width
