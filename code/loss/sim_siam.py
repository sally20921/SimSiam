import math
# from operator import itemgetter
from torch import nn
from torch.nn.functional as F
from torch.nn.modules.loss import _Loss

from metric.stat_metric import StatMetric

# y_pred : {'p_i': , 'p_j': , 'z_i': , 'z_j': }
# return loss.item(), stats, batch_size, y_pred.detach(), target.detach()
class SimSiamLoss(object):
    def __init__(self, use_outputs, loss_metric):
        super().__init__()

        self.net_output_key = use_outputs
        self.reduction = loss_metric

#    @staticmethod
#    def get_metric():
#        return {'sim_siam_loss': StatMetric(output_transform=lambda x: x[1]['sim_siam_loss'], x[2])} # loss.item(), batch_size

    def _loss(p,z):
        z = z.detach() # stop gradient
        p = F.normalize(p, dim=1) # l2-normalize
        z = F.normalize(z, dim=1) # l2-normalize
        return -(p*z).sum(dim=1).mean()

    # return type torch.Tensor
    def loss(p_i, p_j, z_i, z_j):
        return _loss(p_i,z_j) / 2 + _loss(p_j, z_i) / 2

    def forward(self, y_pred):
        # y_pred is a dictionary
        p_i, p_j, z_i, z_j = (v for k, v in sorted(y_pred.items()))
        loss = _loss(p_i,z_j) / 2 + _loss(p_j, z_i) / 2
        loss = self._reduce(loss) 
        return loss # {'sim_siam_loss': loss.item()} # tensor, {'':python number} 

    # mean or sum torch.Tensor t into one element tensor
    # if loss is not one element
    def _reduce(self, t):
        func = {
            'none': lambda x: x,
            'mean': lambda x: x.mean(),
            'sum': lambda x: x.sum(),
            'max': lambda x: x.max()
        }[self.reduction]

        return func(t)
    
    @classmethod
    def resolve_args(cls, args):
        return cls(use_outputs=args.use_outputs, loss_metric=args.loss_metric)


