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


from collections import namedtuple

Genotype = namedtuple(
    'Genotype',
    ['normal', 'normal_concat', 'reduce', 'reduce_concat',
     'alphas_normal', 'alphas_reduce'])

PRIMITIVES = None


def set_primitives(num_ops):
    primitives = [
        'skip_connect',
        'sep_conv_3x3',
        'max_pool_3x3',
        'dil_conv_3x3',
        'sep_conv_5x5',
        'avg_pool_3x3',
        'sep_conv_7x7',
    ]

    global PRIMITIVES
    if num_ops > 0:
        PRIMITIVES = primitives[:num_ops]
        out = primitives[:num_ops]
    else:
        PRIMITIVES = primitives[:]
        out = primitives[:]

    return out


# av: added UNAS cells on different datasets
UNAS_CIFAR10 = Genotype(normal=[[{'inputs': [0, 0], 'ops': [4, 0]},
                                 {'inputs': [0, 1], 'ops': [0, 3]},
                                 {'inputs': [0, 1], 'ops': [1, 3]},
                                 {'inputs': [1, 0], 'ops': [1, 3]}]],
                        normal_concat=range(2, 6),
                        reduce=[[{'inputs': [0, 0], 'ops': [1, 4]},
                                 {'inputs': [1, 0], 'ops': [0, 1]},
                                 {'inputs': [1, 0], 'ops': [5, 0]},
                                 {'inputs': [3, 1], 'ops': [4, 5]}]],
                        reduce_concat=range(2, 6),
                        alphas_normal=None,
                        alphas_reduce=None)

UNAS_CIFAR100 = Genotype(normal=[[{'inputs': [0, 0], 'ops': [1, 3]},
                                  {'inputs': [1, 0], 'ops': [3, 1]},
                                  {'inputs': [0, 1], 'ops': [0, 1]},
                                  {'inputs': [1, 0], 'ops': [4, 1]}]],
                         normal_concat=range(2, 6),
                         reduce=[[{'inputs': [0, 0], 'ops': [6, 1]},
                                  {'inputs': [1, 0], 'ops': [1, 4]},
                                  {'inputs': [1, 0], 'ops': [6, 3]},
                                  {'inputs': [0, 3], 'ops': [3, 1]}]],
                         reduce_concat=range(2, 6),
                         alphas_normal=None,
                         alphas_reduce=None)

UNAS_IMAGENET = Genotype(normal=[[{'inputs': [0, 0], 'ops': [1, 0]},
                                  {'inputs': [0, 1], 'ops': [1, 0]},
                                  {'inputs': [1, 0], 'ops': [1, 4]},
                                  {'inputs': [0, 1], 'ops': [1, 3]}]],
                         normal_concat=range(2, 6),
                         reduce=[[{'inputs': [0, 1], 'ops': [4, 1]},
                                  {'inputs': [0, 1], 'ops': [4, 1]},
                                  {'inputs': [1, 0], 'ops': [4, 3]},
                                  {'inputs': [3, 1], 'ops': [3, 1]}]],
                         reduce_concat=range(2, 6),
                         alphas_normal=None,
                         alphas_reduce=None)
# av end.