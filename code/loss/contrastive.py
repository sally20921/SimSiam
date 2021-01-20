import math
from torch import nn
from torch.nn.functional as F
from torch.nn.modules.loss import _Loss

import copy 
import random
from functools import wraps

# y_pred : {'queries': , 'keys': }
class NTXentLoss(object):
    def __init__(self, args, use_outputs):
        super().__init__()

        self.net_output_key = use_outputs
        self.device = args.device

    def forward(self, y_pred, temp=0.1):
        # y_pred is a dictionary
        keys, queries = (v for k, v in sorted(y_pred.items()))
        b = queries.shape[0]
        logits = queries @ keys.t()
        logits = logits - logits.max(dim=-1,keepdim=True).values
        logits /= temp
        loss = F.cross_entropy(logits, torch.arange(b, device=self.device))
        return loss  # tensor 

    
    @classmethod
    def resolve_args(cls, args):
        return cls(args=args, use_outputs=args.use_outputs)


