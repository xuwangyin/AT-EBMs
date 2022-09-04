import os

import numpy as np
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image
import torch
import random


class TinyImages(Dataset):
    def __init__(self, datafile, transform=None, n_samples=-1):
        super(TinyImages, self).__init__()

        self.data = np.load(datafile)
        if n_samples > 0:
            self.data = self.data[:n_samples]
        assert self.data.shape[1:] == (32, 32, 3)
        assert self.data.dtype == np.uint8

        self.transform = transform

    def __getitem__(self, index):
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(self.data[index])
        if self.transform is not None:
            img = self.transform(img)
        return img

    def __len__(self):
        return self.data.shape[0]

def get_imagenet32_val_dataset(datadir):
    return torchvision.datasets.ImageFolder(
        os.path.join(datadir, 'imagenet256/val'),
        transforms.Compose([transforms.Resize([32,32]), transforms.ToTensor()]))



def get_imagenet256_dataset(datadir, interpolation=2, transform=None):
    return torchvision.datasets.ImageFolder(
        os.path.join(datadir, 'imagenet256/train'),
        transforms.Compose(
            [transforms.RandomResizedCrop(256, interpolation=interpolation),
             transforms.RandomHorizontalFlip(),
             transforms.ToTensor()]))


def get_imagenet256_val_dataset(datadir):
    return torchvision.datasets.ImageFolder(
        os.path.join(datadir, 'imagenet256/val'),
        transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()]))


def get_celebahq256_dataset(datadir, transform=transforms.Compose(
    [transforms.ToTensor()]), attr=None):
    if attr is None:
        return torchvision.datasets.ImageFolder(
            os.path.join(datadir, 'CelebAHQ256/train'), transform)
    else:
        return torchvision.datasets.ImageFolder(
            os.path.join(datadir, f'CelebAHQ256Attr/{attr}'), transform)


def get_afhq256_dataset(datadir, transform=transforms.Compose(
    [transforms.ToTensor()]), subset=None):
    assert subset in [None, 'dog', 'cat', 'wild']
    if subset is None:
        return torchvision.datasets.ImageFolder(
            os.path.join(datadir, 'AFHQ-png/afhq256/train'), transform)
    else:
        return torchvision.datasets.ImageFolder(
            os.path.join(datadir, f'AFHQ-png/afhq256/train/{subset}'),
            transform)


def get_afhq256_val_dataset(datadir, transform=transforms.Compose(
    [transforms.ToTensor()]), subset=None):
    assert subset in [None, 'dog', 'cat', 'wild']
    if subset is None:
        return torchvision.datasets.ImageFolder(
            os.path.join(datadir, 'AFHQ-png/afhq256/val'), transform)
    else:
        return torchvision.datasets.ImageFolder(
            os.path.join(datadir, f'AFHQ-png/afhq256/val/{subset}'), transform)


def get_church256_dataset(datadir, transform=transforms.Compose(
    [transforms.ToTensor()])):
    return torchvision.datasets.ImageFolder(
        os.path.join(datadir, 'Church256/train'), transform)


def get_church256_val_dataset(datadir, transform=transforms.Compose(
    [transforms.ToTensor()])):
    return torchvision.datasets.ImageFolder(
        os.path.join(datadir, 'Church256/test'), transform)


