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

from collections import namedtuple

Genotype = namedtuple(
    'Genotype',
    ['normal', 'reduce', 'alphas_normal', 'alphas_reduce'])

BLOCK_PRIMITIVES = [
    'mic_zero',
    'mic_3x3_e3',
    'mic_3x3_e6',
    'mic_5x5_e3',
    'mic_5x5_e6',
    'mic_7x7_e3',
    'mic_7x7_e6',
]

REDUCTION_PRIMITIVES = [
    'mic_3x3_e3',
    'mic_3x3_e6',
    'mic_5x5_e3',
    'mic_5x5_e6',
    'mic_7x7_e3',
    'mic_7x7_e6',
]


PROXYLESS_GPU = Genotype(
    normal=[
        'mic_zero',    # 2.
        'mic_zero',    # 3.
        'mic_zero',    # 4.
        'mic_zero',    # 6.
        'mic_zero',    # 7.
        'mic_3x3_e3',  # 8.
        'mic_zero',    # 10.
        'mic_zero',    # 11.
        'mic_5x5_e3',  # 12.
        'mic_zero',    # 14.
        'mic_3x3_e3',  # 15.
        'mic_5x5_e3',  # 16.
        'mic_7x7_e6',  # 18.
        'mic_7x7_e6',  # 19.
        'mic_5x5_e6',  # 20.
    ],
    reduce=[
        'mic_5x5_e3',  # 1.
        'mic_7x7_e3',  # 5.
        'mic_7x7_e6',  # 9.
        'mic_5x5_e6',  # 13.
        'mic_7x7_e6',  # 17.
        'mic_7x7_e6',  # 21.
    ],
    alphas_normal=None,
    alphas_reduce=None
)

# cell discovered by UNAS in the ProxylessNAS search space.
UNAS = Genotype(
    normal=[
        'mic_3x3_e3',
        'mic_zero',
        'mic_zero',
        'mic_zero',
        'mic_zero',
        'mic_zero',
        'mic_3x3_e3',
        'mic_3x3_e3',
        'mic_3x3_e3',
        'mic_3x3_e3',
        'mic_3x3_e3',
        'mic_3x3_e3',
        'mic_7x7_e3',
        'mic_7x7_e3',
        'mic_7x7_e3'
    ],
    reduce=[
        'mic_5x5_e3',
        'mic_7x7_e3',
        'mic_7x7_e3',
        'mic_5x5_e3',
        'mic_7x7_e6',
        'mic_7x7_e6'
    ],
    alphas_normal=None,
    alphas_reduce=None
)