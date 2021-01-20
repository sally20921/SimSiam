from pathlib import Path
import random

from fire import Fire
from munch import Munch
# A Munch is a Python dictionary that provides attribute-style access

import torch
import numpy as np

from config import pretrain, linear_eval, fine_tune
from dataloader.load_dataset import get_dataset
from dataloader import get_aug
from utils import wait_for_key, suppress_stdout
from train import train
#from evaluate import evaluate
#from infer import infer 

class Cli:
    def __init__(self):
        self.pretrain = pretrain
        self.linear_eval = linear_eval
        self.fine_tune = fine_tune
        self.debug = debug_options

    def _default_args(self, **kwargs):
        args = [self.pretrain, self.linear_eval, self.fine_tune]
        for arg in args:
            # update arguments if any argument was given from command
            args.update(kwargs)

            # update arguments to its absolute path
            args.update(resolve_paths(arg))
            args.update(fix_seed(arg))
            args.update(get_device(arg))

            # at the end, print all the args
            print(arg)

        return Munch(self.pretrain), Munch(self.linear_eval), Munch(self.fine_tune)

    # the most important part, checking dataloader dir
    def check_dataloader(self, **kwargs):
        from dataloader.load_dataset import modes
        from utils import prepare_batch
        from tqdm import tqdm 
        print("check_dataloader")

        pretrain, linear_eval, fine_tune = self._default_args(**kwargs) 
        ptaug = get_aug(args=args, train=True, double=True)
        pvaug = get_aug(args=args, train=True, double=False)
        pretrain_iters = get_dataset(args, ptaug, pvaug)

        ltaug = get_aug(args, True, False)
        lvaug = get_aug(args, False, False)
        linear_iters = get_dataset(args, ltaug, lvaug)

        ftaug = get_aug(args, True, False)
        fvaug = get_aug(args, False, False)
        fine_tune = get_dataset(args, ftaug, fvaug)

        # Get a batch of training data
        train_iter_test = next(iter(pretrain_iters['train'])) 
        # see if iteration in train proceeds well
        for (key1, key2), value in train_iter_test.items():
            if isinstance(value, torch.Tensor):
                print(key1, key2, value.shape)
            else:
                print(key1, key2, value)
        
        #for mode in modes:
        #    print('Test loading %s data' % mode)
        #    for batch_idx, batch in tqdm(pretrain_iters[mode]):
        #    #    import ipdb; ipdb.set_trace() # XXX DEBUG
        #        batch = prepare_batch(args, batch)

    def pretrain(self, **kwargs):
        args, _, _ = self._default_args(**kwargs)
        train(args)
        wait_for_key()

    def linear_eval(self, **kwargs):
        _, args, _ = self._default_args(**kwargs)
        linear_eval(args)
        wait_for_key()

    def fine_tune(self, **kwargs):
        _, _, args = self._default_args(**kwargs)
        fine_tune(args)
        wait_for_key()

def resolve_paths(config):
    paths = [k for k in config.keys() if k.endswith('_path')]
    res  = {}
    root = Path('../').resolve()
    for path in paths:
        res[path] = root / config[path]

    return res

def fix_seed(args):
    if 'random_seed' not in args:
        args['random_seed'] = 0
    random.seed(args['random_seed'])
    np.random.seed(args['random_seed'])
    torch.manual_seed(args['random_seed'])
    torch.cuda.manual_seed_all(args['random_seed'])
    return args

def get_device(args):
    if hasattr(args, 'device'):
        device = args.device

    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return {'device': device}

if __name__ == "__main__":
    Fire(Cli)
