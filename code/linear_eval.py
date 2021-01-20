import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from tqdm import tqdm

from ignite.engine.engine import Engine, State, Events
from ignite.metrics import Accuracy, TopKCategoricalAccuracy

from ckpt import get_model_ckpt
from model import get_model
from loss import get_loss
from logger import log_results_cmd
from dataloader import get_aug
from dataloader import get_dataset
from dataloader.load_dataset import get_iterator
from metric.stat_metric import StatMetric
from optimizer import get_optimizer, get_scheduler

from utils import prepare_batch

def get_evaluator(args, model, loss_fn, metrics={}):
    from termcolor import colored

    sample_count = 0

    def _inference(evaluator, batch):
        nonlocal sample_count

        model.eval()
        classifier.train()
        with torch.no_grad():
            net_inputs, target = prepare_batch(args, batch)
            feature = model(**net_inputs)
            y_pred = classifier(feature)
            batch_size = y_pred.shape[0]
            loss, stats = loss_fn(y_pred, target)

            return loss.item(), stats, batch_size, y_pred, target 

    engine = Engine(_inference)

    metrics = {**metrics, **{
        'loss': StatMetric(ouput_transform=lambda x: (x[0], x[2])),
        'top1_acc': StatMetric(output_transform=lambda x: ((x[3].argmax(dim=-1) == x[4]).float().mean().item(), x[2]))
    }}
    if hasattr(loss_fn, 'get_metric'):
        metrics = {**metrics, **loss_fn.get_metric()}

    for name, metric in metric.items():
        metric.attach(engine, name)

    return engine


def evaluate(args):
    print(args)
    transforms = get_aug(args)
    train_loader, test_loader = get_dataset(args, transforms)
    
    _, model, iters, ckpt_available = get_model_ckpt(args)
    classifier = get_classifier(args, model)

    loss_fn = get_loss(args)
    optimizer = get_optimizer(args)
    scheduler = get_scheduler(args, optimizer)

    metrics = get_metrics(args)
    evaluator = get_evaluator(args, model, loss_fn, metrics)

    state = evaluate_once(evaluator, iterator=iters['val'])
    log_results_cmd('valid/epoch', state, 0)

