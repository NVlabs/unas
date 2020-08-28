# ---------------------------------------------------------------
# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for UNAS. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

import torch
import torch.nn as nn


class SurrogateLinear(nn.Module):
    def __init__(self, alpha_size, logging):
        super(SurrogateLinear, self).__init__()
        self.layer = nn.Linear(alpha_size, 1)
        self.logging = logging

    def forward(self, alphas):
        return self.layer(alphas)

    def learn(self, train_alphas, train_target, test_alphas, test_target):
        train_alphas = torch.cat([train_alphas, torch.ones(train_alphas.size(0), 1)], dim=1)
        test_alphas = torch.cat([test_alphas, torch.ones(test_alphas.size(0), 1)], dim=1)
        w = torch.inverse(torch.mm(train_alphas.transpose(0, 1), train_alphas) + 1.0 * torch.eye(train_alphas.size(1)))
        w = torch.mm(w, train_alphas.transpose(0, 1))
        w = torch.mm(w, train_target.view(-1, 1))
        train_loss = torch.mean(torch.abs(torch.mm(train_alphas, w) - train_target.view(-1, 1)))
        test_loss = torch.mean(torch.abs(torch.mm(test_alphas, w) - test_target.view(-1, 1)))

        self.logging.info('surrogate, train loss %f, test loss %f' % (train_loss, test_loss))
        print('surrogate, train loss %f, test loss %f, mean target %f' % (train_loss, test_loss, float(torch.mean(train_target))))

        bias = w[-1, :]
        w = w[:-1, :]
        self.layer.weight.data = w.transpose(0, 1).cuda().clone()
        self.layer.bias.data = bias.cuda().clone()