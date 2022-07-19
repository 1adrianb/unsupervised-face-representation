import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, projection_size, depth=2, add_bn=False):
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
        if add_bn:
            layers.append(nn.BatchNorm1d(projection_size))
        self.net = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.net(x)

class SimSiamNet(nn.Module):
    def __init__(self, base_model, dim=2048):
        super(SimSiamNet, self).__init__()

        # create the encoders
        self.base_net = base_model
        out_size = self.base_net.base_width * 8 * self.base_net.expansion
        
        self.projector = MLP(out_size, hidden_size=dim, projection_size=dim, depth=3, add_bn=True)
        self.predictor = MLP(dim, hidden_size=dim//4, projection_size=dim)

        self.reset_parameters()

    def reset_parameters(self):
        # reset conv initialization to default uniform initialization
        for m in self.modules():
            print(m)
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
                stdv = 1. / math.sqrt(n)
                m.weight.data.uniform_(-stdv, stdv)
                if m.bias is not None:
                    m.bias.data.uniform_(-stdv, stdv)
            elif isinstance(m, nn.Linear):
                stdv = 1. / math.sqrt(m.weight.size(1))
                m.weight.data.uniform_(-stdv, stdv)
                if m.bias is not None:
                    m.bias.data.uniform_(-stdv, stdv)

    def _distance(self, p, z):
        #return - F.cosine_similarity(p, z.detach(), dim=-1).mean()
        #z = z.detach()
        
        p = F.normalize(p, dim=1)
        z = F.normalize(z, dim=1)
        return -(p*z.detach()).sum(dim=1).mean()

    # def forward(self, images):
    #     x = torch.cat(images, dim=0)
    #     z = self.projector(self.base_net(x))
    #     p = self.predictor(z)
        
    #     z1, z2 = torch.split(z, z.size(0)//2, dim=0)
    #     p1, p2 = torch.split(p, p.size(0)//2, dim=0)
        
    #     loss = self._distance(p1, z2)/2 + self._distance(p2, z1)/2

    #     return loss

    def forward(self, images):
        z1 = self.projector(self.base_net(images[0]))
        z2 = self.projector(self.base_net(images[1]))

        p1, p2 = self.predictor(z1), self.predictor(z2)
        
        loss = self._distance(p1, z2) / 2 + self._distance(p2, z1) / 2

        return loss

class SimSiam(object):
    def __init__(self, model, args):
        super().__init__()
        self.model = model
        self.args = args

        self.model = model
        self.queue = None

    def on_epoch_start(self, epoch):
        pass

    def on_iter_end(self, epoch, iter_number, num_batches):
        pass
        
    def __call__(self, inputs):
        return self.model(inputs)