from utils.buffer import Buffer
from torch.nn import functional as F
from models.utils.continual_model import ContinualModel
from utils.args import *
import torch
from copy import deepcopy


def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Continual learning via'
                                        ' Dark Experience Replay++.')
    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)
    parser.add_argument('--alpha', type=float, required=True,
                        help='Penalty weight.')
    parser.add_argument('--beta', type=float, required=True,
                        help='Penalty weight.')
    return parser


class TAM(ContinualModel):
    NAME = 'tam'
    COMPATIBILITY = ['class-il', 'task-il']

    def __init__(self, backbone, loss, args, transform):
        super(TAM, self).__init__(backbone, loss, args, transform)

        self.buffer = Buffer(self.args.buffer_size, self.device)
        self.task_id = 0
        self.exclude_layers_start_with = ['ae', 'linear']
        self.ema_model = deepcopy(self.net).to(self.device)
        # set regularization weight
        self.reg_weight = self.args.reg_weight
        # set parameters for ema model
        self.ema_update_freq = self.args.ema_update_freq
        self.ema_alpha = self.args.ema_alpha
        self.consistency_loss = torch.nn.MSELoss(reduction='none')
        self.current_task = 0
        self.global_step = 0

    def update_ema_model_variables(self):
        alpha = min(1 - 1 / (self.global_step + 1), self.ema_alpha)
        for ema_param, param in zip(self.ema_model.parameters(), self.net.parameters()):
            ema_param.data.mul_(alpha).add_(1 - alpha, param.data)

    def end_task(self, dataset):
        self.task_id += 1

    def buffer_through_ae(self):
        buf_inputs, buf_labels, task_labels = self.buffer.get_data(
            self.args.minibatch_size, transform=self.transform)
        buf_feats = self.net.features(buf_inputs)
        buf_feats_ema = self.ema_model.features(buf_inputs)
        buf_outout_ae = torch.zeros((self.args.minibatch_size, self.task_id + 1, buf_feats.shape[-1]),
                                      device=self.device)
        buf_outout_ae_ema = torch.zeros((self.args.minibatch_size, self.task_id + 1, buf_feats.shape[-1]),
                                      device=self.device)
        err_ae_1 = torch.zeros((self.args.minibatch_size, self.task_id + 1), device=self.device)
        for i in range(self.task_id + 1):
            out_ae_i = self.net.ae[i](buf_feats)
            out_ae_i_copy = self.ema_model.ae[i](buf_feats_ema.detach())
            recon_e = F.mse_loss(out_ae_i, buf_feats, reduction='none')
            err_ae_1[:, i] = torch.mean(recon_e, dim=1)
            buf_outout_ae[:, i, :] = out_ae_i
            buf_outout_ae_ema[:, i, :] = out_ae_i_copy

        # current model
        indices = torch.argmin(err_ae_1, dim=1)
        mask = F.one_hot(indices, self.task_id + 1)
        mask = mask.unsqueeze(2).expand(-1, -1, buf_feats.shape[-1])
        buf_outout_ae = torch.sum(buf_outout_ae * mask, keepdim=True, dim=1).squeeze()
        buf_outputs = self.net.linear(buf_feats * buf_outout_ae)
        # EMA model
        mask_ema = F.one_hot(task_labels.long(), self.task_id + 1)
        mask_ema = mask_ema.unsqueeze(2).expand(-1, -1, buf_feats_ema.shape[-1])
        buf_outout_ae_ema = torch.sum(buf_outout_ae_ema * mask_ema, keepdim=True, dim=1).squeeze()
        buf_logits = self.ema_model.linear(buf_feats_ema * buf_outout_ae_ema)

        return buf_outputs, buf_logits, buf_labels

    def observe(self, inputs, labels, not_aug_inputs):
        self.opt.zero_grad()
        # outputs = self.net(inputs)
        # CE for current task samples
        loss = 0
        feats = self.net.features(inputs)
        outputs_encoder = self.net.ae[self.task_id].encoder(feats)
        outputs_ae = self.net.ae[self.task_id].decoder(outputs_encoder)
        outputs = self.net.linear(outputs_ae * feats)
        loss += self.loss(outputs, labels)

        if self.args.use_pairwise_loss_after_ae or self.args.load_best_args:
            softmax = torch.nn.Softmax(dim=-1)
            for i in range(self.task_id):
                outputs_i = self.net.ae[i](feats)
                pairwise_dist = torch.pairwise_distance(softmax(outputs_i.detach()),
                                                        softmax(outputs_ae), p=1).mean()
                loss -= self.args.pairwise_weight * (pairwise_dist)

        loss_1 = torch.tensor(0)
        if not self.buffer.is_empty():
            buf_outputs_1, buf_logits_1, _ = self.buffer_through_ae()
            loss_1 = self.reg_weight * F.mse_loss(buf_outputs_1, buf_logits_1.detach())
            loss += loss_1

            # CE for buffered images
            buf_outputs_2, _, buf_labels_2 = self.buffer_through_ae()
            loss += self.args.beta * self.loss(buf_outputs_2, buf_labels_2)

        loss.backward()
        self.opt.step()

        task_labels = torch.ones(labels.shape[0], device=self.device) * self.task_id
        self.buffer.add_data(examples=not_aug_inputs,
                             labels=labels,
                             task_labels=task_labels)

        # Update the ema model
        self.global_step += 1
        if torch.rand(1) < self.ema_update_freq:
            self.update_ema_model_variables()

        return loss.item(), loss_1.item()
