import math
import os
from collections import defaultdict

import PIL
from PIL import Image, ImageOps
from tqdm import tqdm

import torch
from torchvision import datasets, transforms
from torchvision.transforms import GaussianBlur
import torchvision.transforms as T

from utils import *
from .vision import VisionDataset


transform_types = ['moco', 'sim_siam', 'byol', 'eval', 'simclr', 'swav']

# image_size = # imagenet 224 # cifar 32
delimiter = '/'
# two_crops_transform = True

imagenet_norm = [[0.485, 0.456, 0.406],[0.229, 0.224, 0.225]]
cifar_norm = [[0.4914, 0.4822, 0.4465],[0.2023, 0.1994, 0.2010]]

class StandardTransform:
    def __init__(self, image_size=224, train=True, double=True, normalize=imagenet_norm):
        self.double = double 
        self.train = train 

        if train == True:
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(image_size, scale=(0.08, 1.0), ratio=(3.0/4.0,4.0/3.0), interpolation=Image.BICUBIC),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(*normalize)
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(int(image_size*(8/7)), interpolation=Image.BICUBIC), # 224 -> 256 
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(*normalize)
            ])

    def __call__(self, x):
        if double == True:
            return self.transform(x), self.transform(x)
        else:
            return self.transform(x)

    @classmethod
    def resolve_args(cls, args, train, double):
        image_size = args.get("image_size", 224)
        normalize = args.get("normalize")
        return cls(image_size=image_size, train=train, double=double, normalize=normalize)


class BYOLTransform:
    def __init__(self, image_size=224, train=True, double=True, normalize=imagenet_norm):
        self.double = double
        self.train = train 

        self.transform1 = transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.08, 1.0), ratio=(3.0/4.0,4.0/3.0), interpolation=Image.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([transforms.ColorJitter(0.4,0.4,0.2,0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.GaussianBlur(kernel_size=image_size//20*2+1, sigma=(0.1, 2.0)), # simclr paper gives the kernel size. Kernel size has to be odd positive number with torchvision
            transforms.ToTensor(),
            transforms.Normalize(*normalize)
        ])
        self.transform2 = transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.08, 1.0), ratio=(3.0/4.0,4.0/3.0), interpolation=Image.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([transforms.ColorJitter(0.4,0.4,0.2,0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            # transforms.RandomApply([GaussianBlur(kernel_size=int(0.1 * image_size))], p=0.1),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=image_size//20*2+1, sigma=(0.1, 2.0))], p=0.1),
            transforms.RandomApply([Solarization()], p=0.2),
            
            transforms.ToTensor(),
            transforms.Normalize(*normalize)
        ])
        self.train_transform = transforms.Compose([
                transforms.RandomResizedCrop(image_size, scale=(0.08, 1.0), ratio=(3.0/4.0,4.0/3.0), interpolation=Image.BICUBIC),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(*normalize)
            ])

        self.val_transform = transforms.Compose([
                transforms.Resize(int(image_size*(8/7)), interpolation=Image.BICUBIC), # 224 -> 256 
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(*normalize)
            ])

    def __call__(self, x):
        if self.train and self.double:
            return self.transform1(x), self.transform2(x)
        elif self.train:
            return self.train_transform(x)
        else:
            return self.val_transform(x)

class SimCLRTransform:
    def __init__(self, image_size=224, train=True, double=True, normalize=imagenet_norm):
        s = 1.0
        self.double = double
        self.train = train
        self.transform = T.Compose([
            T.RandomResizedCrop(image_size, scale=(0.2, 1.0)),
            T.RandomHorizontalFlip(),
            T.RandomApply([T.ColorJitter(0.8*s,0.8*s,0.8*s,0.2*s)], p=0.8),
            T.RandomGrayscale(p=0.2),
            T.RandomApply([T.GaussianBlur(kernel_size=image_size//20*2+1, sigma=(0.1, 2.0))], p=0.5),
            # We blur the image 50% of the time using a Gaussian kernel. We randomly sample σ ∈ [0.1, 2.0], and the kernel size is set to be 10% of the image height/width.
            T.ToTensor(),
            T.Normalize(*mean_std)
        ])

    def __call__(self, x):
        if self.double:
            return self.transform(x), self.transform(x)
        else:
            return self.transform(x)


class SimSiamTransform:
    def __init__(self, image_size=224, train=True, double=True, normalize=imagenet_norm):
        p_blur = 0.5 if image_size > 32 else 0
        self.train = train
        self.double = double
        self.transform = T.Compose([
            T.RandomResizedCrop(image_size, scale=(0.2, 1.0)),
            T.RandomHorizontalFlip(),
            T.RandomApply([T.ColorJitter(0.4,0.4,0.4,0.1)], p=0.8),
            T.RandomGrayscale(p=0.2),
            T.RandomApply([T.GaussianBlur(kernel_size=image_size//20*2+1, sigma=(0.1, 2.0))], p=p_blur),
            T.ToTensor(),
            T.Normalize(*mean_std)
        ])

    def __call__(self, x):
        if self.double:
            return self.transform(x), self.transform(x)
        else:
            return self.transform(x)





