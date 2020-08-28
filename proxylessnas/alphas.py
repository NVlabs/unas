# ---------------------------------------------------------------
# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for UNAS. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from proxylessnas.genotypes import Genotype
from gsm_utils import gumbel_softmax_sample


class Alpha(nn.Module):
    def __init__(self, num_normal, num_reduce, num_op_normal, num_op_reduce, gsm_soften_eps, gsm_temperature, gsm_type,
                 same_alpha_minibatch):
        super(Alpha, self).__init__()
        self.num_normal = num_normal
        self.num_reduce = num_reduce
        self.num_op_normal = num_op_normal
        self.num_op_reduce = num_op_reduce
        self._gsm_soften_eps = gsm_soften_eps
        self._temperature = gsm_temperature
        self._same_alpha_minibatch = same_alpha_minibatch
        self.gsm_type = gsm_type
        self._initialize_alphas()

    def forward(self, batch_size):
        def _sample_weights(alpha):
            if self._gsm_soften_eps > 0.:
                q = F.softmax(alpha, dim=-1)
                alpha = torch.log(q + self._gsm_soften_eps)

            alpha = alpha.unsqueeze(1)
            if not self._same_alpha_minibatch:
                alpha = alpha.expand(-1, batch_size, -1)

            weights = gumbel_softmax_sample(alpha, self._temperature, self.gsm_type)
            return weights

        def _expand_alpha(alpha):
            if self._gsm_soften_eps > 0.:
                q = F.softmax(alpha, dim=-1)
                alpha = torch.log(q + self._gsm_soften_eps)

            alpha = alpha.unsqueeze(1)
            if not self._same_alpha_minibatch:
                alpha = alpha.expand(-1, batch_size, -1)

            return alpha

        weights_normal = _sample_weights(self.alphas_normal)
        weights_reduce = _sample_weights(self.alphas_reduce)

        discrete_normal = F.one_hot(torch.argmax(weights_normal.clone(), dim=-1),
                                    weights_normal.size(-1)).detach().float().cuda()
        discrete_reduce = F.one_hot(torch.argmax(weights_reduce.clone(), dim=-1),
                                    weights_reduce.size(-1)).detach().float().cuda()

        alpha_normal = _expand_alpha(self.alphas_normal)
        alpha_reduce = _expand_alpha(self.alphas_reduce)

        return {'normal': weights_normal, 'reduce': weights_reduce,
                'logit_normal': alpha_normal, 'logit_reduce': alpha_reduce,
                'dis_normal': discrete_normal, 'dis_reduce': discrete_reduce}

    def clone_weights(self, weights):
        weights_normal = weights['normal'].clone().detach()
        weights_reduce = weights['reduce'].clone().detach()
        return {'normal': weights_normal, 'reduce': weights_reduce}

    def _initialize_alphas(self):
        self.alphas_normal, self.alphas_reduce = self.initialize_architecture_params()
        self.arch_parameters = []
        self.arch_parameters.append(self.alphas_normal)
        self.arch_parameters.append(self.alphas_reduce)
        self.arch_parameters = nn.ParameterList(self.arch_parameters)

    def initialize_architecture_params(self):
        alphas_normal = nn.Parameter(torch.zeros(self.num_normal, self.num_op_normal).cuda(), requires_grad=True)
        alphas_reduce = nn.Parameter(torch.zeros(self.num_reduce, self.num_op_reduce).cuda(), requires_grad=True)

        return alphas_normal, alphas_reduce

    def alphas_size(self):
        normal_size = self.num_normal * self.num_op_normal
        reduce_size = self.num_reduce * self.num_op_reduce
        return normal_size, reduce_size

    def genotype(self, normal_primitives, reduce_primitives):
        """Stores connections information for current cell."""

        def _parse_proxyless(weights, primitives):
            # Find the best op in this weight.
            k_best = np.argmax(weights, axis=1)
            return [primitives[k] for k in k_best]

        _parse = _parse_proxyless
        gene_normal = _parse(F.softmax(self.alphas_normal, dim=-1).data.cpu().numpy(), normal_primitives)
        gene_reduce = _parse(F.softmax(self.alphas_reduce, dim=-1).data.cpu().numpy(), reduce_primitives)
        alphas_normal = self.alphas_normal
        alphas_reduce = self.alphas_reduce
        genotype = Genotype(
            normal=gene_normal,
            reduce=gene_reduce,
            alphas_normal=alphas_normal,
            alphas_reduce=alphas_reduce,
        )
        return genotype

    def get_arch_sample(self, weights):
        # optimize surrogate network
        alphas = torch.cat([weights['normal'].view(1, -1), weights['reduce'].view(1, -1)], dim=1)
        return alphas

    def log_prob(self, weights):
        def log_q(d, a):
            return torch.sum(torch.sum(d * a, dim=-1) - torch.logsumexp(a, dim=-1), dim=0)

        return log_q(weights['dis_normal'], weights['logit_normal']) + \
               log_q(weights['dis_reduce'], weights['logit_reduce'])

    def entropy_loss(self, weights):
        def ent(logit):
            q = F.softmax(logit, dim=-1)
            return - torch.sum(torch.sum(q * logit, dim=-1) - torch.logsumexp(logit, dim=-1), dim=0)
        e = ent(weights['logit_normal']) + ent(weights['logit_reduce'])
        return torch.mean(e)

    def discretize(self, weights):
        return {'normal': weights['dis_normal'], 'reduce': weights['dis_reduce']}

    def plot_alphas(self, ops, is_normal, title=''):
        fig, ax = plt.subplots()
        fig.set_figheight(10)
        fig.set_figwidth(7)

        if is_normal:
            prob = F.softmax(self.alphas_normal, dim=-1)
        else:
            prob = F.softmax(self.alphas_reduce, dim=-1)
        alpha = prob.data.cpu().numpy()

        num_edges = alpha.shape[0]

        ax.xaxis.tick_top()
        plt.imshow(alpha, vmin=0, vmax=1)
        plt.xticks(range(len(ops)), ops)
        plt.xticks(rotation=30)
        plt.yticks(range(num_edges), range(1, num_edges + 1))
        for i in range(num_edges):
            for j in range(len(ops)):
                val = alpha[i][j]
                val = '%.4f' % (val)
                ax.text(j, i, val, va='center',
                        ha='center', color='white', fontsize=8)

        plt.colorbar()
        plt.tight_layout()
        fig.suptitle(title, fontsize=16, fontweight='bold')

        return fig
