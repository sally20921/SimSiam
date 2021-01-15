import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange

'''
baseline model
'''
class Baseline(nn.Module):
    def __init__(self, layers, dropout):
        super().__init__()

        self.layers = layers
        self.dropout = dropout

    @classmethod
    def resolve_args(cls, args):
        return cls(args.layers, args.dropout)

    def forward(self, x_i, x_j):
        return x_i, x_j


