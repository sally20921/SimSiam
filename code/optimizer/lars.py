''' 
SimCLR is optimized using LARS with linear learning rate scaling
(i.e. LearningRate = 0.3 * BatchSize / 256)
(i.e. weight decay = 1e-6)
(i.e. linear warmup for the first 10 epochs)
(i.e. lr_decay with the cosine decay schedule without restarts)
'''

from torch import optim
import torch
from torch.optim.optimizer import Optimizer, required

class LARS(optim.SGD):
    @classmethod
    def resolve_args(cls, args, params):
        options = {}
        options['lr'] = args.get("learning_rate", 0.01)
        #options['lr_decay'] = args.get("lr_decay", 0)
        options['weight_decay'] = args.get("weight_decay", 0)
        options['momentum'] = args.get("momentum", 0.9)
        return cls(params, **options)

