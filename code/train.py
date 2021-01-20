''' torch.nn.modules.modules 
class Module:
    vartype training : bool 
    def __init__(self):
        self.training = True
        self._parameters = OrderedDict()

    def train(self: T, mode:bool = True)
    # sets the module in training mode. e.g. class: Dropout, class: BatchNorm
        self.training = mode
        for module in self.chilren():
            module.train(mode)
        return self
'''

from ignite.engine.engine import Engine, State, Events

from ckpt import get_model_ckpt, save_ckpt
from loss import get_loss
from optimizer import get_sub_optimizer, get_optimizer, get_scheduler # (args, model) # (args, optimizer) # (args, optimizer)
from logger import get_logger, log_results, log_results_cmd

from utils import prepare_batch
from metric import get_metrics
from evaluate import get_evaluator, evaluate_once
from ignite.metrics import Accuracy, TopKCategoricalAccuracy
from metric.stat_metric import StatMetric, KNNMonitor

import numpy as np

def get_trainer(args, model, loss_fn, optimizer):
    def update_model(trainer, batch):
        # set model to training mode
        model.train(True)
        optimizer.zero_grad()
        # to GPU prepare batch
        net_inputs, target = prepare_batch(args, batch)
        # **: dictionary into each argument
        # y_pred : dict {z_i, z_j, p_i, p_j}
        y_pred = model(**net_inputs)
        batch_size = target.shape[0] # N
        loss, stats = loss_fn(y_pred)
        loss = loss.mean() # differential dynamic programming
        loss.backward()
        optimizer.step()
        scheduler.step() 
        return loss.item(), stats, batch_size, y_pred.detach()

'''
torch.Tensor
detach() : returns a new Tensor, detached from the current graph. The result will never require a gradient.
item(): returns a value of this tensor as a standard Python number . This only works for tensors with one element. 
'''
    trainer = Engine(update_model)
   
    '''
    ignite.metrics.Accuracy(output_transform: Callable = <function Accuracy.<lambda>>, is_multilabel: bool = False, device=Optional[Union[str, torch.device]]=None)
    # calculates the accuracy for binary, multiclass and multilabel data
    # `update` must receive output of the form `(y_pred, y)` or `{'y_pred': y_pred, 'y': y}
    # `y_pred` must be in the following shape (batch_size, num_categories, ...) or (batch_size, ...)
    # `y` must be in the following shape (batch_size, ...)
    # `y` and `y_pred` must be in the following shape of (batch_size, num_categories, ...) and  `num_categories` must be greater than 1 for multilabel cases.
    # In binary and multilabel cases, the elements of `y` and `y_pred` should have 0 or 1 values.
    def thresholded_output_transform(output):
        y_pred, y = output
        y_pred = torch.round(y_pred)
        return y_pred, y
    binary_accuracy = Accuracy(threshold_output_transform)
'''
    metrics = {
        'Accuracy': KNNMonitor(output_transform=lambda x:x[3]),
        'Top-5 Accuracy': TopKCategoricalAccuracy(k=5),
            } # loss is same as simsiam_loss 

    if hasattr(loss_fn, 'get_metric'):
        metrics = {**metrics, **loss_fn.get_metric()}

    for name, metric in metrics.items():
        metric.attach(trainer, name)

    return trainer

def train(args):
    args, model, iters, ckpt_available = get_model_ckpt(args)

    if ckpt_available:
        print("loaded checkpoint {}".format(args.ckpt_name))
    loss_fn = get_loss(args)
    sub_optimizer = get_sub_optimizer(args, model)
    optimizer = get_optimizer(args, sub_optimizer)
    scheduler = get_scheduler(args, optimizer)

    trainer = get_trainer(args, model, loss_fn, optimizer)

    metrics = get_metrics(args)
    evaluator = get_evaluator(args, model, loss_fn, metrics)

    logger = get_logger(args)
    
    @trainer.on(Events.STARTED)
    def on_training_started(engine):
        print("Begin Training")
    
    # log batch-wise
    @trainer.on(Events.ITERATION_COMPLETED)
    def log_iter_results(engine):
        log_results(logger, 'train/iter', engine.state, engine.state.iteration)
    
    # epoch-wise valid eval + ckpt
    @trainer.on(Events.EPOCH_COMPLETED)
    def evaluate_epoch(engine):
        log_results(logger, 'train.epoch', engine.state, engine.state.epoch)
        state = evaluate_once(evaluator, iterator=iters['val'])
        log_results(logger, 'valid/epoch', state, engine.state.epoch)
        log_results_cmd('valid/epoch', state, engine.state.epoch)
        save_ckpt(args, engine.state.epoch, engine.state.metrics['sim_siam_loss'], model)

    trainer.run(iters['train']), max_epochs=args.max_epoch)


