from collections import defaultdict
import math
import os
from collections import defaultdict

import PIL
from PIL import Image, ImageOps
from tqdm import tqdm

import torch
import torchvision
from torchvision import datasets, transforms
from torchvision.transforms import GaussianBlur

from utils import *

dataset_types = ['tinyimagenet','mnist', 'stl10', 'cifar10', 'cifar100', 'imagenet', 'random']
imagenet_norm = [[0.485, 0.456, 0.406],[0.229, 0.224, 0.225]]
cifar_norm = [[0.4914, 0.4822, 0.4465],[0.2023, 0.1994, 0.2010]]

def get_dataset(args, train_transform, val_transform):
    if args.datasets == 'mnist':
        trainloader = torchvision.datasets.MNIST(args.image_path, train=True, transform=train_transform, download=True)
        valloader = torchvision.datasets.MNIST(args.image_path, train=False, transform=val_transform, download=True)

    elif args.dataset == 'stl10':
        trainloader = torchvision.datasets.STL10(args.image_path, split='train+unlabeled', transform = train_transform, download=True)
        valloader = torchvision.datasets.STL10(args.image_path, split='test', transform=val_transform, download=True)

    elif args.dataset == 'cifar10':
        trainloader = torchvision.datasets.CIFAR10(args.image_path, train=True, transform=train_transform, download=True)
        valloader = torchvision.datasets.CIFAR10(args.image_path, train=False, transform=val_transform, download=True)

    elif args.dataset == 'cifar100':
        trainloader = torchvision.datasets.CIFAR100(args.image_path, train=True, transform=train_transform, download=True)
        valloader = torchvision.datasets.CIFAR100(args.image_path, train=False, transform=val_transform, download=True)
    
    elif args.dataset == "imagenet":
        # assume imagenet dataset already exists, because it takes too long to download imagenet data
        traindir = os.path.join(args.image_path, 'train')
        valdir = os.path.join(args.image_path, 'val')
        trainloader = torchvision.datasets.ImageFolder(traindir, transform=train_transform)
        valloader = torchvision.datasets.ImageFolder(valdir, transform=val_transform)
    elif args.dataset == "tinyimagenet":
        '''
        imagenet2012subset is a subset of original imagenet dataset.
        The dataset share the same validation set as the original.
        The training set is subsampled in a label balanced fashion.
        This dataset requires you to download the source data manually.
        manual_dir should contain two files: ILSVRC2012_img_train.tar 
        and ILSVRC2012_img_val.tar
        http://www.image-net.org/download-images
        '''
    else:
        raise NotImplementedError

    train_loader = torch.utils.data.DataLoader(trainloader, batch_size=args.batch_sizes[0], shuffle=args.shuffle[0], num_workers=args.num_workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(valloader, batch_size=args.batch_sizes[1], shuffle=args.shuffle[1], num_workers=args.num_workers, pin_memory=True)
    return {'train_iter': train_loader, 'val_iter': val_loader, 'test_iter': val_loader}
