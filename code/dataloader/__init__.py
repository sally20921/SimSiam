import os
from pathlib import Path

import torch
import torchvision
from torchvision import transforms
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset

from inflection import underscore

dataset_dict = {}
'''
class.__mro__
a tuple of classes that are considered when looking for base classes during method resolution
'''
def add_datasets():
    path = Path(os.path.dirname(__file__))

    for p in path.glob('*.py'):
        name = p.stem
        parent = p.parent.stem
        if name != "__init__":
            __import__("{}.{}".format(parent, name))
            module = eval(name)
            for member in dir(module):
                member = getattr(module, member)
                if (inspect.isclass(member) and \
                        str(member.__name__).endswith('Dataset')) or \
                        (if inspect.isclass(member) and \
                        (str(member.__name__).endswith('Transform') and 'transforms' in member.__dict__.keys())):
                            dataset_dict[underscore(str(member.__name__))] = member # contains either Dataset or Transform Class

#def get_dataset(args, transforms):
#    dataset = dataset_dict[args.datasets]
#    dataset = dataset.resolve_args(args, transforms)
#    return dataset # return train_iter for train, val_iter for val etc.

def get_aug(args, train, double):
    aug = dataset_dict[args.transforms]
    aug = aug.resolve_args(args, train, double)
    return aug

add_datasets()

