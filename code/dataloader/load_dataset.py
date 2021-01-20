from collections import defaultdict

import torch

from utils import *
from .preprocess_image import preprocess_images

import os
import re
from tqdm import tqdm
import numpy as np

from torch.utils.data import Dataset, DataLoader

modes = ['train', 'val', 'test']

class BaseDataset:
    def __init__(self, args, train_test_transforms):
        self.config = args
        self.train_test_transforms = train_test_transforms
        self.dataset = self.load_data()
 
    def load_data(self):
        raise NotImplementedError

    def __call__:
        return self.dataset

class ImageNetDataset(BaseDataset):
    def __init__(self, args, train_test_transforms):
        super().__init__(self, args=args, train_test_transforms=train_test_transforms)

    def load_data(self):
        args = self.config
        if args["stage"] == 'pretrain':
            print("Loading pretraining data")
            traindir = os.path.join(args["image_path"], 'train')
            trainloader = torch.utils.data.DataLoader()
            return trainloader

        else: # args["stage"] == 'linear_eval':
            print("Loading evaluation data")
            valdir = os.path.join(args["image_path"], 'val')
            valloader = torch.utils.data.DataLoader()
            return valloader

def load_data(args):
    print('Loading image data')

    train_iter, traintest_iter, val_iter = preprocess_images(args)

    return {'train': train_iter, 'traintest': traintest_iter, 'val': val_iter, 'test': val_iter}

def get_iterator(args):
    iters = load_data(args)
    print("Data loading done")

    return iters



