import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision 
from tqdm import tqdm
import shutil
import copy

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

def get_dataloader(args, pretrain_model, train_loader, test_loader):
    train_x, train_y = compute_features(args, train_loader, pretrain_model)
    test_x, test_y = compute_features(args, test_loader, pretrain_model)

    train_dataset = torch.utils.data.TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size[0], shuffle=args.shuffle[0])

    test_dataset = torch.utils.data.TensorDataset(torch.from_numpy(x_test), torch.from_numpy(y_test))
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size[1], shuffle=args.shuffle[1]))

    return train_loader, test_loader

def compute_features(args, loader, pt_model):
    features = []
    labels = []

    for step, batch in enumerate(loader):
        x, y = _prepare_batch(args, batch)
        with torch.no_grad():
            y_pred = pt_model(x)
        
        h = y_pred['h'].detach()

        features.extend(h.cpu().detach().numpy())
        labels.extend(y.numpy())

    features = np.array(features)
    labels = np.array(labels)

def get_trainer(args, model, loss_fn, optimizer):
    def update_model(trainer, batch):
        model.train()
        optimizer.zero_grad()

        # to gpu
        net_inputs, target = prepare_batch(args, batch)
        # **: dictionary input each argument
        # y_pred: dict {z_i, z_j, p_i, p_j}
        y_pred = model(**net_inputs)
        batch_size = target.shape[0] # N
        loss = loss_fn(y_pred, target)
        
        loss.backward()
        optimizer.step()
        
        return loss.item(), batch_size, y_pred.detach(), target.detach()

    trainer = Engine(update_model)

    metrics = {
            'loss': StatMetric(output_transform=lambda x: (x[0], x[1])),
            'top1_acc': Accuracy(output_transform=lambda x: (x[2], x[3])),
            }

    for name, metric in metrics.items():
        metric.attach(trainer, name)

    return trainer
def get_evaluator(args, model, loss_fn, metrics={}):
    sample_count = 0
    def _inference(evaluator, batch):
        nonlocal sample_count
        
        model.eval()
        with torch.no_grad():
            net_inputs, target = prepare_batch(args, batch)
            y_pred = model(**net_inputs)
            batch_size = y_pred.shape[0]
            loss = loss_fn(y_pred, target)

            return loss.item(), batch_size, y_pred, target

        engine= Engine(_inference)

        metrics = {**metrics, **{
            'loss': StatMetric(ouput_transform=lambda x: (x[0], x[1])),
            'top1_acc': Accuracy(output_transform=lambda x: (x[2], x[3])),
        }}

        for name, metric in metrics.items():
            metric.attach(engine, name)

        return engine

def evaluate_once(evaluator, iterator):
    evaluator.run(iterator)
    return evaluator.state

def logistic_regression(pretrain, args):
    # get pretrained models
    args, pt_model, ckpt_available = get_model_ckpt(pretrain)
    pt_model.eval()

    taug = get_aug(args=args, train=True, double=False)
    vaug = get_aug(args=args, train=False, double=False)
    log_iters = get_dataset(args, taug, vaug)
    log_iters = get_dataloader(args, pt_model, log_iters['train'], log_iters['test'])

    if ckpt_available:
        print("loaded checkpoint {}".format(args.ckpt_name))
    
    model = get_model(args, pt_model.num_features, args.num_classes)
    loss_fn = get_loss(args)
    optimizer = get_sub_optimizer(args, model)
    
    # create features from pretrained model

    trainer = get_trainer(args, model, loss_fn, optimizer)
    evaluator = get_evaluator(args, model, loss_fn)

    metrics = get_metrics(args)

    logger = get_logger(args)

    @trainer.on(Events.STARTED)
    def on_training_started(engine):
        print("Begin Logistic Regression")

    # batch-wise
    @trainer.on(Events.ITERATION_COMPLETED)
    def log_iter_results(engine):
        log_results(logger, 'logistic_regression/iter', engine.state, engine.state.iteration)

    # epoch-wise (ckpt)
    @trainer.on(Events.EPOCH_COMPLETED)
    def save_epoch(engine):
        log_results(logger, 'logistic_regression/epoch', engine.state, engine.state.epoch)
        log_results_cmd(logger, 'logistic_regression/epoch', engine.state, engine.state.epoch)
        state = evaluate_once(evaluator, iterator=log_iters['val'])
        save_ckpt(args, engine.state.epoch, engine.state.metrics['loss'], model)

    trainer.run(log_iters['train']), max_epochs=args.max_epochs)
