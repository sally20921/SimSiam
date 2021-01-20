import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision 
from tqdm import tqdm

from ignite.engine.engine import Engine, State, Events

from dataloader import get_aug
from dataloader.load_dataset import get_dataset
from ckpt import get_model_ckpt, save_ckpt
from model import get_model
from loss import get_loss
from optimizer import get_optimizer, get_sub_optimizer, get_scheduler
from logger import get_logger, log_results, log_results_cmd

from utils import prepare_batch
from metric import get_metrics
from metric.stat_metric import StatMetric, KNNMonitor
from ignite.metrics import Accuracy, TopKCategoricalAccuracy
import numpy as np

def get_trainer(args, model, loss_fn, optimizer, scheduler):
    def update_model(trainer, batch):
        model.train()
        optimizer.zero_grad()

        # to gpu
        net_inputs, target = prepare_batch(args, batch)
        # **: dictionary input each argument
        # y_pred: dict {z_i, z_j, p_i, p_j}
        y_pred = model(**net_inputs)
        batch_size = target.shape[0] # N
        loss = loss_fn(y_pred)
        loss = loss.mean() #dpp
        
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        return loss.item(), batch_size, y_pred.detach()

    trainer = Engine(update_model)

    metrics = {
            'loss': StatMetric(output_transform=lambda x: (x[0], x[1])),
            }

    for name, metric in metrics.items():
        metric.attach(trainer, name)

    return trainer

def pretrain(args):
    args, model, ckpt_available = get_model_ckpt(args)
    
    taug = get_aug(args=args, train=True, double=True)
    vaug = get_aug(args=args, train=False, double=True)
    pretrain_iters = get_dataset(args, taug, vaug)

    if ckpt_available:
        print("loaded checkpoint {}".format(args.ckpt_name))
    loss_fn = get_loss(args)
    sub_optimizer = get_sub_optimizer(args, model)
    optimizer = get_optimizer(args, sub_optimizer)
    scheduler = get_scheduler(args, optimizer)

    trainer = get_trainer(args, model, loss_fn, optimizer, scheduler)

    metrics = get_metrics(args)

    logger = get_logger(args)

    @trainer.on(Events.STARTED)
    def on_training_started(engine):
        print("Begin Pretraining")

    # batch-wise
    @trainer.on(Events.ITERATION_COMPLETED)
    def log_iter_results(engine):
        log_results(logger, 'pretrain/iter', engine.state, engine.state.iteration)

    # epoch-wise (ckpt)
    @trainer.on(Events.EPOCH_COMPLETED)
    def save_epoch(engine):
        log_results(logger, 'pretrain/epoch', engine.state, engine.state.epoch)
        log_results_cmd(logger, 'pretrain/epoch', engine.state, engine.state.epoch)
        save_ckpt(args, engine.state.epoch, engine.state.metrics['loss'], model)

    trainer.run(pretrain_iters['train']), max_epochs=args.max_epochs)
