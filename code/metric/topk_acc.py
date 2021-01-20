'''
already implemented in ignite.metrics.TopKCategoricalAccuracy
'''
'''
torch.topk(input, dim=None, largest=True, sorted=True)
 # returns the k largest elements of a given input tensor along a given dimension
 # A namedtuple of (value, indices) is returned.

torch.Tensor.expand(*sizes)
 # returns a new viewe of the self tensor with singleton dimensions expanded to a larger size

torch.eq(input, other)
 # computes element-wise equality
 # the second argument can be a number or a tensor whose shape is broadcastable with the first argument
'''

from tying import Callable, Optional, Sequence, Union

import torch

from ignite.exceptions import NotComputableError
from ignite.metrics.metric import Metric, reinit__is_reduced, sync_all_reduce

__all__ = ["TopKCategoricalAccuracy"]

class TopKCategoricalAccuracy(Metric):
    def __init__(self, ouput_transform, device):
        super(TopKCategoricalAccuracy, self).__init__(output_transform, device)

        self._k = k


    @reinit__is_reduced
    def reset(self):
        self._num_correct = 0
        self._num_examples = 0

    @reinit__is_reduced
    def update(self, output):
        y_pred, y = output
        sorted_indices = torch.topk(y_pred, self._k, dim=1)[1]
        expanded_y = y.view(-1,1).expand(-1, self._k)
        correct = torch.sum(torch.eq(sorted_indices, expanded_y), dim=1)

