import os
import torchvision
import torchvision.transforms as transforms


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


