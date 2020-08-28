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
from collections import OrderedDict
from darts.genotypes import Genotype, PRIMITIVES
from gsm_utils import gumbel_softmax_sample


class Alpha(nn.Module):
    def __init__(self, num_normal, num_reduce, num_op, num_nodes, gsm_soften_eps, gsm_temperature, gsm_type, same_alpha_minibatch):
        super(Alpha, self).__init__()
        self.num_normal = num_normal
        self.num_reduce = num_reduce
        self.num_op = num_op
        self.num_nodes = num_nodes
        self._gsm_soften_eps = gsm_soften_eps
        self._temperature = gsm_temperature
        self._same_alpha_minibatch = same_alpha_minibatch
        self.gsm_type = gsm_type
        self._initialize_alphas()

    def forward(self, batch_size):
        def _sample_weights(alphas, step):
            weights_cont = {}
            weights_disc = {}
            all_alpha = {}
            for key in alphas:
                alpha = alphas[key][step]
                # soften weights
                if self._gsm_soften_eps > 0.:
                    q = F.softmax(alpha, dim=-1)
                    alpha = torch.log(q + self._gsm_soften_eps)

                alpha = alpha.unsqueeze(1)
                if not self._same_alpha_minibatch:
                    alpha = alpha.expand(-1, batch_size, -1)

                w = gumbel_softmax_sample(alpha, self._temperature, self.gsm_type)
                weights_cont[key] = w
                weights_disc[key] = F.softmax(1000 * w.clone().detach(), dim=-1)
                all_alpha[key] = alpha

            b = weights_cont['input'].size(1)
            weights_cont['input_op0'] = (
                    weights_cont['input'][0].view(b, -1, 1) * weights_cont['op'][0].view(b, 1, -1))
            weights_cont['input_op1'] = (
                    weights_cont['input'][1].view(b, -1, 1) * weights_cont['op'][1].view(b, 1, -1))
            weights_disc['input_op0'] = (
                    weights_disc['input'][0].view(b, -1, 1) * weights_disc['op'][0].view(b, 1, -1))
            weights_disc['input_op1'] = (
                    weights_disc['input'][1].view(b, -1, 1) * weights_disc['op'][1].view(b, 1, -1))

            return weights_cont, weights_disc, all_alpha

        all_weights_normal_cont, all_weights_reduce_cont = [], []
        all_weights_normal_disc, all_weights_reduce_disc = [], []
        all_alphas_normal, all_alphas_reduce = [], []

        for alpha_normal_layer in self.alphas_normal:
            weights_normal_cont, weights_normal_disc, alphas_normal = [], [], []
            for step in range(self.num_nodes):
                weight_normal_cont, weight_normal_disc, alpha_normal = _sample_weights(alpha_normal_layer, step)
                weights_normal_cont.append(weight_normal_cont)
                weights_normal_disc.append(weight_normal_disc)
                alphas_normal.append(alpha_normal)

            all_weights_normal_cont.append(weights_normal_cont)
            all_weights_normal_disc.append(weights_normal_disc)
            all_alphas_normal.append(alphas_normal)

        for alpha_reduce_layer in self.alphas_reduce:
            weights_reduce_cont, weights_reduce_disc, alphas_reduce = [], [], []
            for step in range(self.num_nodes):
                weight_reduce_cont, weight_reduce_disc, alpha_reduce = _sample_weights(alpha_reduce_layer, step)
                weights_reduce_cont.append(weight_reduce_cont)
                weights_reduce_disc.append(weight_reduce_disc)
                alphas_reduce.append(alpha_reduce)

            all_weights_reduce_cont.append(weights_reduce_cont)
            all_weights_reduce_disc.append(weights_reduce_disc)
            all_alphas_reduce.append(alphas_reduce)

        return {'normal': all_weights_normal_cont, 'reduce': all_weights_reduce_cont,
                'dis_normal': all_weights_normal_disc, 'dis_reduce': all_weights_reduce_disc,
                'logit_normal': all_alphas_normal, 'logit_reduce': all_alphas_reduce}

    def clone_weights(self, weights):
        def _clone_weights(weights):
            cloned_weight_all_layers = []
            for weights_per_layer in weights:
                cloned_weight = []
                for w in weights_per_layer:
                    cloned_w = {}
                    for k, v in w.items():
                        cloned_w[k] = v.clone().detach()
                    cloned_weight.append(cloned_w)
                cloned_weight_all_layers.append(cloned_weight)
            return cloned_weight_all_layers

        return {'normal': _clone_weights(weights['normal']),
                'reduce': _clone_weights(weights['reduce'])}

    def _initialize_alphas(self):
        self.alphas_normal, self.alphas_reduce = self.initialize_architecture_params()
        self._arch_parameters = []
        for alpha_normal in self.alphas_normal:
            for k, v in alpha_normal.items():
                self._arch_parameters.extend(v)
        for alpha_reduce in self.alphas_reduce:
            for k, v in alpha_reduce.items():
                self._arch_parameters.extend(v)
        self._arch_parameters = nn.ParameterList(self._arch_parameters)

    def initialize_architecture_params(self):
        num_ops = self.num_op

        def create_alphas():
            input_selector = []
            op_selector = []

            for i in range(self.num_nodes):
                input_selector.append(nn.Parameter(torch.zeros(size=[2, i + 2]).cuda(), requires_grad=True))
                op_selector.append(nn.Parameter(torch.zeros(size=[2, num_ops]).cuda(), requires_grad=True))

            out = OrderedDict()
            out['input'] = input_selector
            out['op'] = op_selector
            return out

        alphas_normal = [create_alphas() for _ in range(self.num_normal)]
        alphas_reduce = [create_alphas() for _ in range(self.num_reduce)]

        return alphas_normal, alphas_reduce

    def arch_parameters(self):
        return self._arch_parameters

    def genotype(self):
        """Stores connections information for current cell."""
        def _parse(alphas):
            """Does the Paired Input selection procedure.
            2 incoming edges per node are selected and
            the best op per edge is retained.
            """
            genes = []
            for alpha in alphas:
                gene = []
                for step in range(self.num_nodes):
                    inputs = alpha['input'][step].data.cpu().numpy()
                    ops = alpha['op'][step].data.cpu().numpy()
                    top_inputs = np.argmax(inputs, axis=1)
                    top_ops = np.argmax(ops, axis=1)

                    gene.append({
                        'inputs': top_inputs,
                        'ops': top_ops,
                    })
                genes.append(gene)
            return genes

        gene_normal = _parse(self.alphas_normal)
        gene_reduce = _parse(self.alphas_reduce)
        concat = range(2, self.num_nodes + 2)
        genotypes = [
            Genotype(
                normal=gene_normal, normal_concat=concat,
                reduce=gene_reduce, reduce_concat=concat,
                alphas_normal=self.alphas_normal,
                alphas_reduce=self.alphas_reduce,
            )
        ]
        return genotypes

    def get_arch_sample(self, weights):
        # optimize surrogate network
        alphas = []
        for l in range(len(weights['normal'])):
            for i in range(len(weights['normal'][l])):
                alphas.append(weights['normal'][l][i]['input_op0'])
                alphas.append(weights['normal'][l][i]['input_op1'])
        for l in range(len(weights['reduce'])):
            for i in range(len(weights['reduce'][l])):
                alphas.append(weights['reduce'][l][i]['input_op0'])
                alphas.append(weights['reduce'][l][i]['input_op1'])

        alphas = torch.cat(alphas, dim=1).view(alphas[0].size(0), -1)

        return alphas

    def log_prob(self, weights):
        def log_q(d, a):
            return torch.sum(torch.sum(d * a, dim=-1) - torch.logsumexp(a, dim=-1), dim=0)

        log_q_d = 0
        for l in range(len(weights['logit_normal'])):
            num_steps = len(weights['logit_normal'][l])
            for i in range(num_steps):
                for key in weights['logit_normal'][l][i].keys():
                    a = weights['logit_normal'][l][i][key]
                    w = weights['dis_normal'][l][i][key]
                    log_q_d += log_q(w, a)
        for l in range(len(weights['logit_reduce'])):
            num_steps = len(weights['logit_reduce'][l])
            for i in range(num_steps):
                for key in weights['logit_reduce'][l][i].keys():
                    a = weights['logit_reduce'][l][i][key]
                    w = weights['dis_reduce'][l][i][key]
                    log_q_d += log_q(w, a)

        return log_q_d

    def entropy_loss(self, weights):
        def ent(logit):
            q = F.softmax(logit, dim=-1)
            return - torch.sum(torch.sum(q * logit, dim=-1) - torch.logsumexp(logit, dim=-1), dim=0)

        log_q_d = 0
        for l in range(len(weights['logit_normal'])):
            for i in range(len(weights['logit_normal'][l])):
                for key in weights['logit_normal'][l][i].keys():
                    logit = weights['logit_normal'][l][i][key]
                    log_q_d += ent(logit)
        for l in range(len(weights['logit_reduce'])):
            for i in range(len(weights['logit_reduce'][l])):
                for key in weights['logit_reduce'][l][i].keys():
                    logit = weights['logit_reduce'][l][i][key]
                    log_q_d += ent(logit)

        log_q_d = torch.mean(log_q_d, dim=0)
        return log_q_d

    def discretize(self, weights):
        return {'normal': weights['dis_normal'], 'reduce': weights['dis_reduce']}

    def plot_alphas(self, weights):
        fig, ax = plt.subplots(1, 2)
        fig.set_figheight(6)
        fig.set_figwidth(13)

        for idx, k in enumerate(sorted(weights.keys())):
            prob = [F.softmax(a, dim=-1) for a in weights[k]]
            if k == 'input':
                selector = torch.cat(prob, 1)
            elif k == 'combiner':
                selector = torch.cat(prob, 0).view(len(weights[k]), -1)
            else:
                selector = torch.cat(prob, 0)

            selector = selector.data.cpu().numpy()
            im = ax[idx].imshow(selector, vmin=0, vmax=1)
            # ax[idx].set_title(k)
            ax[idx].set_xlabel(k)
            xticks = self.get_xticks(weights[k], k)
            plt.sca(ax[idx])
            ax[idx].xaxis.tick_top()
            plt.xticks(range(len(xticks)), xticks)
            plt.xticks(ha='left', rotation=30, fontsize=8)
            plt.yticks(range(selector.shape[0]), range(1, selector.shape[0] + 1))

        cbaxes = fig.add_axes([0.05, 0.2, 0.01, 0.6])
        fig.colorbar(im, cax=cbaxes)
        cbaxes.yaxis.set_ticks_position('left')

        return fig

    def get_xticks(self, weights, key):
        if key == 'input':
            xticks = []
            for i in range(len(weights)):
                for j in range(i + 2):
                    xticks.append(j)
        elif key == 'op':
            xticks = PRIMITIVES
        else:
            raise NotImplementedError

        return xticks

    def alpha_loss(self, weights):
        loss = 0
        all_weights = weights['normal'] + weights['reduce']
        steps = self.num_nodes
        for w_layer in all_weights:
            for w in w_layer[:steps]:
                loss += torch.sum(w['input_op0'] * w['input_op1'], dim=[1, 2])

        return loss