import math

from torch import nn
from torch.nn.functional as F
from torch.nn.modules.loss import _Loss

from metric.stat_metric import StatMetric

# input could be ((z_1, p_1), (z_2, p_2))

class SimSiamLoss(object):
    def __init__(self):
        super().__init__()

    def forward(self, p, z):
        z = z.detach() # stop gradient
        p = F.normalize(p, dim=1) # l2-normalize
        z = F.normalize(z, dim=1) # l2-normalize
        return -(p * z).sum(dim=1).mean()

