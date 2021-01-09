from pathlib import Path
import random

from fire import Fire
from munch import Munch

import torch
import numpy as np

from config import config, debug_options
from dataloader.load_dataset import get_iterator
from utils import wait_for_key, suppress_stdout
from train import train
from evaluate import evaluate
from infer import infer 

class Cli:
    def __init__(self):
        self.defaults = config
        self.debug = debug_options

    # update arguments if any argument was given from command
    def _default_args(self, **kwargs):
        args = self.defaults
        if 'debug' in kwargs:
            args.update(self.debug)
        args.update(kwargs)
        
        #update arguments to its absolute path
        args.update(resolve_path(config))
        args.update(fix_seed(args))
        args.update(get_device(args))
        
        # at the end, print all the updated arguments
        print(args)

        return Munch(args)

    # the most important part, checking dataloader dir
    def check_dataloader(self, **kwargs):
        from dataloader.load_dataset import modes
        from utils import prepare_batch
        from tqdm import tqdm 

        args = self._default_args(**kwargs)
        iters = get_iterator(args)

        train_iter_test = next(iter(iters['train']))
        # see if train to test proceeds well
        for key, value in train_iter_test.items():
            if isinstance(value, torch.Tensor):
                print(key, value.shape)
            else:
                print(key, value)



