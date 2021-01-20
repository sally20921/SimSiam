import math
# from operator import itemgetter
from torch import nn
from torch.nn.functional as F
from torch.nn.modules.loss import _Loss

# from metric.stat_metric import StatMetric

# y_pred : {'z_i': , 'z_j': }
class NTXentLoss(object):
    def __init__(self, args, use_outputs):
        super().__init__()

        self.net_output_key = use_outputs
        self.device = args.device

    def forward(self, y_pred, temp=0.5):
        # y_pred is a dictionary
        z_i, z_j = (v for k, v in sorted(y_pred.items()))
        z_i = F.normalize(z_i, dim=1)
        z_j = F.normalize(z_j, dim=1)
        N, Z = z_i.shape
        representations = torch.cat([z_i, z_j], dim=0)
        similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=-1)
        l_pos = torch.diag(similarity_matrix, N)
        r_pos = torch.diag(similarity_matrix, -N)

        positives = torch.cat([l_pos, r_pos]).view(2 * N, 1)
        diag = torch.eye(2*N, dtype=torch.bool, self.device)
        diag[N:,:N] = diag[:N,N:] = diag[:N,:N]

        negatives = similarity_matrix[~diag].view(2*N, -1)

        logits = torch.cat([positives, negatives], dim=1)
        logits /= temp

        labels = torch.zeros(2*N, device=self.device, dtype=torch.int64)

        loss = F.cross_entropy(logits, labels, reduction='sum')
        return loss / (2 * N) # tensor, {'':python number} 

    
    @classmethod
    def resolve_args(cls, args):
        return cls(args=args, use_outputs=args.use_outputs)


