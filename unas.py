# ---------------------------------------------------------------
# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for UNAS. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

import time
import os

import numpy as np
import torch
import torch.nn as nn
from apex.fp16_utils import to_python_float
from time import time

from util import utils
from surrogate import SurrogateLinear


class UNAS(object):
    def __init__(self, model, alpha, args, writer, logging):
        self.args = args
        self.model = model
        self.alpha = alpha
        self.logging = logging
        self.arch_optimizer = torch.optim.Adam(self.alpha.parameters(), lr=args.arch_learning_rate,
                                               weight_decay=args.arch_weight_decay)

        self.arch_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.arch_optimizer, float(args.epochs), eta_min=args.arch_learning_rate_min)

        self.latency_cost = args.target_latency > 0.
        self.target_latency = args.target_latency
        if self.latency_cost or self.args.meta_loss == 'relax':
            assert args.meta_loss in {'relax', 'rebar', 'reinforce'}, 'this is only implemented for rebar and reinforce'
            normal_size, reduce_size = self.alpha.module.alphas_size()
            alpha_size = normal_size + reduce_size
            self.surrogate = SurrogateLinear(alpha_size, self.logging).cuda()
            self.latency_pred_loss = utils.AverageMeter()
            self.latency_value = utils.AverageMeter()
            self.latency_coeff = args.latency_coeff
            self.latency_coeff_curr = None
            self.num_repeat = 10
            self.latency_batch_size = 24
            assert self.latency_batch_size <= args.batch_size
            self.num_arch_samples = 10000
            # print('***************** change the number of samples *******')
            # self.num_arch_samples = 200
            self.surrogate_not_train = True

            self.latency_actual = []
            self.latency_estimate = []

        # Extra layers, if any.
        self.meta_loss = args.meta_loss

        # weights generalization error
        self.gen_error_alpha = args.gen_error_alpha
        self.gen_error_alpha_lambda = args.gen_error_alpha_lambda

        # Get the meta learning criterion.
        if self.meta_loss in ['default', 'rebar', 'reinforce']:
            self.criterion = nn.CrossEntropyLoss(reduction='none')
            self.criterion = self.criterion.cuda()

        if self.meta_loss == 'reinforce':
            self.exp_avg1 = utils.ExpMovingAvgrageMeter()
            self.exp_avg2 = utils.ExpMovingAvgrageMeter()

        self.alpha_loss = args.alpha_loss

        # Housekeeping.
        self.loss = None
        self.accuracy = None
        self.count = None
        self.loss_diff_sign = None
        self.reset_counter()
        self.report_freq = args.report_freq
        self.writer = writer

    def reset_counter(self):
        """Resets counters."""
        self.count = 0
        self.loss = utils.AverageMeter()
        self.accuracy = utils.AverageMeter()
        self.loss_diff_sign = utils.AverageMeter()
        if self.latency_cost:
            self.latency_pred_loss = utils.AverageMeter()
            self.latency_value = utils.AverageMeter()

        if self.meta_loss == 'relax':
            self.relax_pred_loss = utils.AverageMeter()

    def mean_accuracy(self):
        """Return mean accuracy."""
        return self.accuracy.avg

    def compute_latency(self, train_batch, discrete_weight):
        discrete_indices = self.model.module.get_indices(discrete_weight)
        self.model.eval()
        train_batch = train_batch[:self.latency_batch_size]
        elapsed_time = np.zeros(self.num_repeat)
        with torch.no_grad():
            for i in range(self.num_repeat):
                torch.cuda.synchronize()
                start = time()
                tmp = self.model.module.fast_forward(train_batch, discrete_indices)
                torch.cuda.synchronize()
                end = time()
                elapsed_time[i] = (end - start)

        self.model.train()
        return np.median(elapsed_time) * 1000

    def train_surrogate(self, train_batch):
        # measure
        self.logging.info('collecting latency samples')
        past_alphas = []
        past_target = []

        for i in range(self.num_arch_samples):
            with torch.no_grad():
                weights = self.alpha(1)
                disc_weights = self.alpha.module.discretize(weights)
                latency = self.compute_latency(train_batch, disc_weights)
                alphas = self.alpha.module.get_arch_sample(disc_weights)
                past_alphas.append(alphas.cpu())
                past_target.append(torch.FloatTensor([latency]))
            if i % 100 == 0:
                self.logging.info('collected %d samples' % i)

        all_alphas = torch.cat(past_alphas, dim=0)
        all_target = torch.cat(past_target, dim=0)

        num_train = int(0.8 * self.num_arch_samples)
        test_alphas = all_alphas[num_train:]
        test_target = all_target[num_train:]
        train_alphas = all_alphas[:num_train]
        train_target = all_target[:num_train]

        self.surrogate.learn(train_alphas, train_target, test_alphas, test_target)
        self.surrogate.eval()
        self.surrogate_not_train = False

    def training_obj(self, train, train_target, weights, model_opt, val, val_target, global_step):
        if not self.gen_error_alpha:
            logits = self.model(train, weights)
            loss = self.criterion(logits, train_target)
            accuracy = utils.accuracy(logits, train_target)[0]
            loss1, loss2 = loss, torch.zeros_like(loss)
        else:
            logits_train = self.model(train, weights)
            loss_train = self.criterion(logits_train, train_target)

            logits_val = self.model(val, weights)
            loss_val = self.criterion(logits_val, val_target)

            loss2 = torch.abs(loss_val - loss_train)
            self.loss_diff_sign.update(torch.mean(((loss_val - loss_train) > 0).float()).data)
            loss1 = loss_train
            loss = loss1 + self.gen_error_alpha_lambda * loss2
            accuracy = utils.accuracy(logits_train, train_target)[0]

        if self.alpha_loss:
            alpha_loss = self.alpha.module.alpha_loss(weights)
            loss += self.args.alpha_loss_lambda * alpha_loss

            if self.count % self.report_freq == 0:
                self.writer.add_scalar(
                    'meta/alpha_loss', torch.mean(alpha_loss), global_step)

        return loss, accuracy, loss1, loss2

    def step(self, input_valid, target_valid, global_step, weights, input_valid2=None, target_valid2=None, model_opt=None):
        """Optimizer for the architecture params."""
        self.arch_optimizer.zero_grad()
        if self.meta_loss == 'default':
            loss, accuracy, loss1, loss2 = self.training_obj(input_valid, target_valid, weights, model_opt,
                                                             input_valid2, target_valid2, global_step)
            loss, loss1, loss2 = torch.mean(loss), torch.mean(loss1), torch.mean(loss2)
        elif self.meta_loss == 'rebar':
            # compute loss with discrete weights
            with torch.no_grad():
                disc_weights = {
                    'normal': weights['dis_normal'], 'reduce': weights['dis_reduce']}

                loss_disc, accuracy, loss1, loss2 = self.training_obj(input_valid, target_valid, disc_weights,
                                                                      model_opt, input_valid2, target_valid2, global_step)

            # compute baseline
            loss_cont, _, _, _ = self.training_obj(input_valid, target_valid, weights,
                                                   model_opt, input_valid2, target_valid2, global_step)

            reward = (loss_disc - loss_cont).detach()
            log_q_d = self.alpha.module.log_prob(weights)
            loss = torch.mean(log_q_d * reward) + torch.mean(loss_cont)
            loss1, loss2 = torch.mean(loss1), torch.mean(loss2)

            if self.latency_cost:
                # train the surrogate function initially.
                if self.surrogate_not_train:
                    self.train_surrogate(input_valid)

                # sample a single architecture sample
                weight_lat = self.alpha(1)
                disc_weights_lat = {'normal': weight_lat['dis_normal'], 'reduce': weight_lat['dis_reduce']}

                # compute latency for the discrete weights.
                elapsed_time = self.compute_latency(input_valid, disc_weights_lat)
                # latency prediction for continuous weights
                self.surrogate.eval()
                alphas = self.alpha.module.get_arch_sample(weight_lat)
                latency_cont = self.surrogate(alphas)
                # latency prediction for discrete weights
                alphas = self.alpha.module.get_arch_sample(disc_weights_lat)
                latency_discrete = self.surrogate(alphas)
                surrogate_loss = torch.mean(torch.abs(elapsed_time - latency_discrete.squeeze(1)))

                self.latency_coeff_curr = self.latency_coeff * max(min(global_step / self.args.latency_iter, 1.0), 0.)
                loss_disc_lat = self.latency_coeff_curr * torch.relu(torch.Tensor([elapsed_time]).cuda() - self.target_latency)
                loss_cont_lat = self.latency_coeff_curr * torch.relu(latency_cont[0] - self.target_latency)

                # collect latency information
                self.latency_pred_loss.update(utils.reduce_tensor(surrogate_loss.data, self.args.world_size))
                self.latency_value.update(elapsed_time)

                self.latency_actual.append(elapsed_time)
                self.latency_estimate.append(latency_discrete.squeeze(1).data.cpu().numpy()[0])

                if global_step % 50 == 0:
                    self.logging.info('latency_pred_loss %f' % np.mean(np.abs(np.array(self.latency_actual)[-50:] - np.array(self.latency_estimate)[-50:])))

                # saving some latency info
                if global_step % 1000 == 100 and self.args.local_rank == 0:
                    import pickle
                    print('saving')
                    with open(os.path.join(self.args.save, 'latency.pkl'), 'wb') as f:
                        pickle.dump([self.latency_actual, self.latency_estimate, global_step], f)

                reward = (loss_disc_lat - loss_cont_lat).detach()
                log_q_d = self.alpha.module.log_prob(weight_lat)
                loss = loss + torch.mean(log_q_d * reward) + torch.mean(loss_cont_lat)

        elif self.meta_loss == 'reinforce':
            # compute loss with discrete weights
            with torch.no_grad():
                disc_weights = self.alpha.module.discretize(weights)
                loss_disc, accuracy, loss1, loss2 = self.training_obj(input_valid, target_valid, disc_weights,
                                                                      model_opt, input_valid2, target_valid2,
                                                                      global_step)

            reduce_loss_disc = utils.reduce_tensor(loss_disc.data, self.args.world_size)
            avg = torch.mean(reduce_loss_disc).detach()
            baseline = self.exp_avg1.avg
            # update the moving average
            self.exp_avg1.update(avg)
            reward = (loss_disc - baseline).detach()
            log_q_d = self.alpha.module.log_prob(weights)
            loss = torch.mean(log_q_d * reward) + baseline
            loss1, loss2 = torch.mean(loss1), torch.mean(loss2)

            if self.latency_cost:
                weight_lat = self.alpha(1)
                disc_weights_lat = self.alpha.module.discretize(weights)
                elapsed_time = self.compute_latency(input_valid, disc_weights_lat)
                self.latency_coeff_curr = self.latency_coeff * min(global_step / self.args.latency_iter, 1.0)
                loss_disc_lat = self.latency_coeff_curr * elapsed_time
                self.latency_value.update(elapsed_time)

                baseline = self.exp_avg2.avg
                # update the moving average
                self.exp_avg2.update(float(loss_disc_lat))
                reward = loss_disc_lat - baseline
                log_q_d = self.alpha.module.log_prob(weight_lat)
                loss = loss + torch.mean(log_q_d * reward) + baseline
                loss1, loss2 = torch.mean(loss1), torch.mean(loss2)

        entropy_loss = self.alpha.module.entropy_loss(weights)

        # Backward pass and update.
        loss.backward()
        self.arch_optimizer.step()
        # Logging.
        reduced_loss = utils.reduce_tensor(loss.data, self.args.world_size)
        accuracy = utils.reduce_tensor(accuracy, self.args.world_size)

        self.loss.update(to_python_float(reduced_loss), 1)
        self.accuracy.update(to_python_float(accuracy), 1)
        self.count += 1
        if self.count % self.report_freq == 0:
            self.logging.info('Meta Loss:%s %03d  %e %f', self.meta_loss,
                              self.count, self.loss.avg, self.accuracy.avg)
            self.writer.add_scalar('meta/loss', self.loss.avg, global_step)
            self.writer.add_scalar('meta/acc', self.accuracy.avg, global_step)
            self.writer.add_scalar('meta/lr', self.arch_optimizer.state_dict()['param_groups'][0]['lr'], global_step)
            self.writer.add_scalar('meta/entropy', entropy_loss, global_step)

            if self.gen_error_alpha:
                self.writer.add_scalar('meta/loss_val', loss1, global_step)
                self.writer.add_scalar('meta/loss_cov', loss2, global_step)
                self.writer.add_scalar('meta/loss_diff_sign', self.loss_diff_sign.avg, global_step)

            if self.latency_cost:
                self.writer.add_scalar('meta/latency_time', self.latency_value.avg, global_step)
                self.writer.add_scalar('meta/latency_prediction_loss', self.latency_pred_loss.avg, global_step)
                self.writer.add_scalar('meta/latency_coeff', self.latency_coeff_curr, global_step)
