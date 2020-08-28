# ---------------------------------------------------------------
# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for UNAS. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

import torch
import torch.nn.functional as F


def sample_gumbel(shape, eps=1e-20):
    """ generate sample from Gumbel distribution. """
    U = torch.Tensor(shape).uniform_(0, 1).cuda()
    sample = -(torch.log(-torch.log(U + eps) + eps))
    return sample


def gumbel_softmax_sample_original(logits, temperature):
    """ generate samples from Gumbel Softmax distribution. """
    y = logits + sample_gumbel(logits.size())
    return F.softmax(y / temperature, dim=-1)


def logsumexp(logits, dim):
    mx = torch.max(logits, dim, keepdim=True)[0]
    return torch.log(torch.sum(torch.exp(logits - mx), dim=dim, keepdim=True)) + mx


def gumbel_softmax_sample_rebar(logits, temperature):
    """ genrate sample from Gumbel Softmax, but add the gradient correction term in Rebar. """
    logits = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
    q = F.softmax(logits, dim=-1)
    u = torch.Tensor(q.size()).uniform_(0, 1).cuda()
    u = torch.clamp(u, 1e-3, 1.-1e-3)

    # draw gsm samples
    z = logits - torch.log(- torch.log(u))
    gsm = F.softmax(z / temperature, dim=-1)

    # compute the correction term for conditional samples
    # see REBAR: https://arxiv.org/pdf/1703.07370.pdf
    k = torch.argmax(z, dim=-1, keepdim=True)
    # get v from u
    u_k = u.gather(-1, k)
    q_k = q.gather(-1, k)
    # This can cause numerical problems, better to work with log(v_k) = u_k / q_k
    # v_k = torch.pow(u_k, 1. / q_k)
    # v.scatter_(-1, k, v_k)
    log_vk = torch.log(u_k) / q_k
    log_v = torch.log(u) - q * log_vk

    # assume k and v are constant
    k = k.detach()
    log_vk = log_vk.detach()
    log_v = log_v.detach()
    g_hat = - torch.log(-log_v/q - log_vk)
    g_hat.scatter(-1, k, -torch.log(- log_vk))
    gsm1 = F.softmax(g_hat / temperature, dim=-1)
    # gsm1 is the correction term in the gradient. Its value is the same as gsm. But its
    # gradient is the same as the correction term in the Eq. 3 of the UNAS paper.
    return gsm - gsm1 + gsm1.detach()


def gumbel_softmax_sample(logits, temperature, gsm_type='original'):
    if gsm_type == 'original':
        return gumbel_softmax_sample_original(logits, temperature)
    elif gsm_type == 'rebar':
        return gumbel_softmax_sample_rebar(logits, temperature)
    else:
        raise NotImplementedError
