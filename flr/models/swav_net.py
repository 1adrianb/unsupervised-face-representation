# Code largely based on: https://github.com/facebookresearch
import os
import copy
import math
from logging import getLogger
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

import numpy as np

logger = getLogger()

def sinkhorn_knopp(Q, world_size, n_iters=3):
    with torch.no_grad():
        Q = shoot_infs(Q)
        sum_Q = torch.sum(Q)
        dist.all_reduce(sum_Q)
        Q /= sum_Q
        
        K, B = Q.size()
        u = torch.zeros(K).cuda(non_blocking=True)
        r = torch.ones(K).cuda(non_blocking=True) / K
        c = torch.ones(B).cuda(non_blocking=True) / (world_size * B)
        
        for i in range(n_iters):
            u = torch.sum(Q, dim=1)
            dist.all_reduce(u)
            u = r / u
            u = shoot_infs(u)
            Q *= u.unsqueeze(1)
            Q *= (c / torch.sum(Q, dim=0)).unsqueeze(0)
            
        return (Q / torch.sum(Q, dim=0, keepdim=True)).t().float()

def shoot_infs(inp_tensor):
    """Replaces inf by maximum of tensor"""
    mask_inf = torch.isinf(inp_tensor)
    ind_inf = torch.nonzero(mask_inf)
    if len(ind_inf) > 0:
        for ind in ind_inf:
            if len(ind) == 2:
                inp_tensor[ind[0], ind[1]] = 0
            elif len(ind) == 1:
                inp_tensor[ind[0]] = 0
        m = torch.max(inp_tensor)
        for ind in ind_inf:
            if len(ind) == 2:
                inp_tensor[ind[0], ind[1]] = m
            elif len(ind) == 1:
                inp_tensor[ind[0]] = m
    return inp_tensor

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
    
class MultiPrototypes(nn.Module):
    def __init__(self, output_dim, num_prototypes):
        super(MultiPrototypes, self).__init__()
        self.num_heads = len(num_prototypes)
        for i, k in enumerate(num_prototypes):
            self.add_module(f'prototypes{i}', nn.Linear(output_dim, k, bias=False))
            
    def forward(self, x):
        for i in range(self.num_heads):
            out.append(getattr(self, f'prototypes{i}')(x))
        return out

class SwAVNet(nn.Module):
    def __init__(self, base_net, projection_hidden_size=4096, projection_size=128, projection_depth=2, num_prototypes=3000):
        super().__init__()
        
        self.base_net = base_net
        out_size = self.base_net.base_width * 8 * self.base_net.expansion
 
        self.projection_net = MLP(out_size, projection_hidden_size, projection_size, depth=projection_depth)

        if isinstance(num_prototypes, list):
            self.prototypes = MultiPrototypes(projection_size, num_prototypes)
        elif num_prototypes > 0:
            self.prototypes = nn.Linear(projection_size, num_prototypes, bias=False)
        
    def forward(self, inputs):
        if not isinstance(inputs, list):
            inputs = [inputs]
        idx_crops = torch.cumsum(torch.unique_consecutive(
            torch.tensor([inp.shape[-1] for inp in inputs]),
            return_counts=True,
        )[1], 0)

        start_idx = 0
        for end_idx in idx_crops:
            _out = self.base_net(torch.cat(inputs[start_idx: end_idx]).cuda(non_blocking=True))
            if start_idx == 0:
                output = _out
            else:
                output = torch.cat((output, _out))
            start_idx = end_idx

        output = self.projection_net(output)
        embedding = F.normalize(output, dim=1, p=2)
        if self.prototypes is not None:
            return embedding, self.prototypes(embedding)
        return embedding

class SwAV(object):
    def __init__(self, model, args):
        super().__init__()
        self.model = model
        self.args = args

        # build the queue
        queue = None
        queue_path = os.path.join(args.dump_path, "queue" + str(args.rank) + ".pth")
        if os.path.isfile(queue_path):
            queue = torch.load(queue_path)["queue"]
            logger.info("Queue loaded succesfuly from {}.".format(queue_path))
        self.queue = queue
        # the queue needs to be divisible by the batch size
        args.queue_length -= args.queue_length % (args.batch_size * args.world_size)

    def on_epoch_start(self, epoch):
        args = self.args
        # optionally starts a queue
        if args.queue_length > 0 and epoch >= args.epoch_queue_starts and self.queue is None:
            self.queue = torch.zeros(
                len(args.crops_for_assign),
                args.queue_length // args.world_size,
                args.feat_dim,
            ).cuda()
        self.use_the_queue = False

    def on_iter_end(self, epoch, iter_number, num_batches):
        pass
        
    def __call__(self, inputs):
        args = self.args
        model = self.model

        # normalize the prototypes
        with torch.no_grad():
            w = model.module.prototypes.weight.data.clone()
            w = F.normalize(w, dim=1, p=2)
            model.module.prototypes.weight.copy_(w)

        # ============ multi-res forward passes ... ============
        embedding, output = model(inputs)
        embedding = embedding.detach()
        bs = inputs[0].size(0)

        # ============ swav loss ... ============
        loss = 0
        for i, crop_id in enumerate(args.crops_for_assign):
            with torch.no_grad():
                out = output[bs * crop_id: bs * (crop_id + 1)]

                # time to use the queue
                if self.queue is not None:
                    if self.use_the_queue or not torch.all(self.queue[i, -1, :] == 0):
                        self.use_the_queue = True
                        out = torch.cat((torch.mm(
                            self.queue[i],
                            model.module.prototypes.weight.t()
                        ), out))
                    # fill the queue
                    self.queue[i, bs:] = self.queue[i, :-bs].clone()
                    self.queue[i, :bs] = embedding[crop_id * bs: (crop_id + 1) * bs]
                # get assignments
                q = out / self.args.epsilon
                if self.args.improve_numerical_stability:
                    M = torch.max(q)
                    dist.all_reduce(M, op=dist.ReduceOp.MAX)
                    q -= M
                q = torch.exp(q).t()
                q = sinkhorn_knopp(q, args.world_size, args.sinkhorn_iterations)[-bs:]

            # cluster assignment prediction
            subloss = 0
            for v in np.delete(np.arange(np.sum(args.nmb_crops)), crop_id):
                p = F.softmax(output[bs * v: bs * (v + 1)] / args.temperature, dim=1)
                subloss -= torch.mean(torch.sum(q * torch.log(p), dim=1))
            loss += subloss / (np.sum(args.nmb_crops) - 1)
        loss /= len(args.crops_for_assign)
        
        return loss

