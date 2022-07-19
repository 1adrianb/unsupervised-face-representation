import os
import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

import numpy as np

def norm_dist(x, y):
    norm_x = torch.norm(x, p=2, dim=-1)
    norm_y = torch.norm(y, p=2, dim=-1)
    return -2 * (x * y).sum(dim=-1) / (norm_x * norm_y)

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, projection_size, depth = 2):
        super().__init__()
        layers = []
        for i in range(depth-1):
            layers += [
                nn.Linear(input_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(inplace=True)
            ]
            input_size = hidden_size
        layers.append(nn.Linear(input_size, projection_size))
        self.net = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.net(x)

class Encoder(nn.Module):
    def __init__(self, base_net, projection_hidden_size=4096, projection_size=128, projection_depth=2):
        super().__init__()
        self.base_net = base_net
        out_size = self.base_net.base_width * 8 * self.base_net.expansion
 
        self.projection_net = MLP(out_size, projection_hidden_size, projection_size, depth=projection_depth)

    def forward(self, x):
        f_features = self.base_net(x).view(x.size(0), -1)
        g_projection = self.projection_net(f_features)    
        return g_projection

class BYOLNet(nn.Module):
    def __init__(self, base_net, projection_hidden_size=4096, projection_size=128, projection_depth=2):
        super().__init__()
        
        self.online_encoder = Encoder(base_net, projection_hidden_size, projection_size, projection_depth)
        self.online_predictor = MLP(projection_size, projection_hidden_size, projection_size)
        self.target_encoder = copy.deepcopy(self.online_encoder)

        # the target is updated via EMA
        for param in self.target_encoder.parameters():
            param.requires_grad = False
        
    def forward(self, inputs):
        assert len(inputs) == 2

        # online encoder
        online_projections = []
        target_projections = []
        for x in inputs:
            online_projection  = self.online_encoder(x)
            online_prediction = self.online_predictor(online_projection)
            online_projections.append(online_prediction)

            with torch.no_grad():
                target_projection = self.target_encoder(x)
                target_projections.append(target_projection)

        loss1 = norm_dist(online_projections[0], target_projections[1].detach())
        loss2 = norm_dist(online_projections[1], target_projections[0].detach())

        return (loss1 + loss2).sum()

class BYOL(object):
    def __init__(self, model, args):
        super().__init__()
        self.model = model
        self.args = args

        self.beta = self.args['beta']
        self.beta_base = self.args['beta_base']

    def _update_beta(self, curr_step, total_steps):
        self.beta = 1 - (1 - self.beta_base) * (math.cos(math.pi*curr_step/total_steps) + 1)/2

    def _update_moving_average(self):
        for curr_params, ma_params in zip(self.model.online_encoder.parameters(), self.target_encoder.parameters()):
            old_weight, curr_weight = ma_params.data, curr_params.data
            if old_weight is None:
                ma_params.data = curr_weight
            else:
                ma_params.data = old_weight * self.beta + (1 - self.beta ) * curr_weight

    def on_epoch_start(self, epoch):
        pass

    def on_iter_end(self, epoch, iter_number, num_batches):
        args = self.args
        model.module._update_beta(iter_number+(epoch*num_batches), args.epochs*num_batches)
        self._update_moving_average()
        
    def __call__(self, inputs):
        loss = self.model(inputs)
        return loss

