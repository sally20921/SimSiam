'''
class Metric(metaclass=ABCMeta):
    def __init__(self, output_transform: Callable = lambda x: x,
    device=None):
        self._output_transform = output_transform
        self._device = device
        self._is_reduced = False
        self.reset()
'''


import torch
from ignite.exceptions import NotComputableError
from ignite.metrics.metric import Metric
from ignite.engine import Events

# epoch-wise stats
class StatMetric(Metric):
    def __init__(self, output_transform=lambda x: x):
        super(StatMetric, self).__init__(output_transform)

        self.log_iter = True

    # triggered every epoch started
    def reset(self):
        self._sum = 0
        slef._num_examples = 0

    # triggered every batch completed
    def update(self, output): # 'simsiam_loss': loss.item(), batch_size
        average_loss = output[0]
        N = output[1]

        self._sum += average_loss * N
        self._num_examples += N
    
    # triggered every epoch completed 
    def compute(self):
        if self._num_examples == 0:
            raise NotComputableError('Loss must have at least one example before it can be computed')
        return self._sum / self._num_examples

    @torch.no_grad()
    def iteration_completed(self, engine, name):
        output = self._output_transform(engine.state.output)
        self.update(output)

        if self.log_iter:
            result = self.compute() #after batch show result up until now
            engine.state.metrics[name] = result

    def attach(self, engine, name):
        # epoch completed with metric name
         engine.add_event_handler(Events.EPOCH_COMPLETED, self.completed, name)

         # epoch started
         if not engine.has_event_handler(self.started, Events.EPOCH_STARTED):
             engine.add_event_handler(Events.EPOCH_STARTED, self.started)
        
        # batch completed with metric name
        if not engine.has_event_handler(self.iteration_completed, Events.ITERATION_COMPLETED):
            engine.add_event_handler(Events.ITERATION_COMPLETED, self.iteration_completed, name)

