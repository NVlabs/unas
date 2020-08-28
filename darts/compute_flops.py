# ---------------------------------------------------------------
# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for UNAS. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

import torch
from ptflops import get_model_complexity_info

from darts.model import NetworkImageNet as Network


def find_flops(genotype, init_channels, layers):
    # Set up the network.
    model = Network(init_channels, 1000, layers, False, genotype)
    model.drop_path_prob = 0

    with torch.cuda.device(0):
        flops, params = get_model_complexity_info(
            model, (3, 224, 224), as_strings=False, print_per_layer_stat=False)
        print('C: %d, Flops: %.2f, Params: %.2f' %
              (init_channels, flops/1e6, params/1e6))
        return flops, params


def find_max_channels(genotype, layers, max_flops):
    best_flops, best_params, best_C = None, None, None
    print(max_flops//1e6)
    if max_flops//1e6 == 600:
        start, end = 30, 70
    elif max_flops//1e6 == 4000:
        start, end = 100, 150
    for C in range(start, end, 2):
        flops, params = find_flops(genotype, C, layers)
        if flops < max_flops:
            best_C = C
            best_flops = flops
            best_params = params
        else:
            break
    return best_flops, best_params, best_C
