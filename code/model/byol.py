import copy
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from math import pi, cos
import math
from .modules import *
from torchvision.models import resnet50

class MLP(nn.Module):
    def __init__(self, args, in_dim, out_dim, hsz, n_layers):
        super(MLP, self).__init__()

        layers = []
        prev_dim = in_dim
        for i in range(n_layers):
            if i == n_layers - 1:
                layers.append(nn.Linear(prev_dim, out_dim))
            else:
                layers.extend([
                    nn.Linear(prev_dim, hsz),
                    nn.BatchNorm1d(hsz, args.eps, args.momentum),
                    nn.ReLU(True)
                ])
                prev_dim = hsz

        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)

class BYOL(nn.Module):
    def __init__(self, args, use_outputs):
        super().__init__()

        self.backbone = resnet50()
        self.projector = MLP(args, resnet50().output_dim, 256, 4096, 2)
        self.online_encoder = nn.Sequential(
                self.backbone,
                self.projector
        )

        self.target_encoder = copy.deepcopy(self.online_encoder)
        self.online_predictor = MLP(256, 256, 4096, 2)

        self.net_output_key = use_outputs
    
    @classmethod
    def resolve_args(cls, args):
        return cls(args, args.use_outputs)

    # {'p_i', 'p_j', 'z_i', 'z_j'}
    def forward(self, x_1, x_2):
        f, h = self.encoder, self.predictor
        z_i, z_j = f(x_1), f(x_2)
        p_i, p_j = h(z_1), h(z_2)
        {key: eval(key) for key in self.net_output_key}
        return y_pred
