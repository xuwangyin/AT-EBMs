import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.datasets
import torchvision.transforms as transforms
from dataset import *
# from pgd_attack import perturb, perturb_random_restarts, perturb_sequence
from pgd_attack import perturb
from sklearn.metrics import roc_curve, auc as auc_fn
from torchvision.utils import make_grid
from misc import set_eval, set_train
from tqdm import tqdm
from pathlib import Path

sys.path.append('./auto-attack')
from autoattack import AutoAttack
sys.path.append('./pytorch-fid/src')
from pytorch_fid.fid_score import calculate_fid_given_paths


def forward(model, x, normalization, which_logit):
    assert which_logit in ['all', 'first', 'max']
    assert normalization in ['cifar10', 'imagenet']
    if normalization == 'cifar10':
        if which_logit in ['first', 'all']:
            # https://github.com/MadryLab/robustness/blob
            # /ca52df73bb94f5a3abb74d95b82a13589354a83e/robustness/datasets
            # .py#L293
            mean = torch.as_tensor([0.4914, 0.4822, 0.4465], dtype=x.dtype,
                                   device=x.device)
            std = torch.as_tensor([0.2023, 0.1994, 0.2010], dtype=x.dtype,
                                  device=x.device)
        elif which_logit == 'max':
            mean = torch.as_tensor(
                [0.4913997551666284, 0.48215855929893703, 0.4465309133731618],
                dtype=x.dtype,
                device=x.device)
            std = torch.as_tensor(
                [0.24703225141799082, 0.24348516474564, 0.26158783926049628],
                dtype=x.dtype,
                device=x.device)
    else:
        if which_logit in ['first', 'all']:
            # Use ImageNet normalization
            # https://pytorch.org/docs/stable/torchvision/models.html
            mean = torch.as_tensor([0.485, 0.456, 0.406], dtype=x.dtype,
                                   device=x.device)
            std = torch.as_tensor([0.229, 0.224, 0.225], dtype=x.dtype,
                                  device=x.device)
        elif which_logit == 'max':
            # RestrictedImageNet normalization
            mean = torch.as_tensor([0.4717, 0.4499, 0.3837], dtype=x.dtype,
                                   device=x.device)
            std = torch.as_tensor([0.2600, 0.2516, 0.2575], dtype=x.dtype,
                                  device=x.device)
    logits = model((x - mean.view(1, 3, 1, 1)) / std.view(1, 3, 1, 1))
    return \
    {'all': logits, 'first': logits[:, 0], 'max': torch.max(logits, dim=1)[0]}[
        which_logit]


class AutoAttackModel(torch.nn.Module):
    """
    A wrapper class to mimic the a two-outputs model for two class
    classificaiton.
    This wrapper is required because AutoAttack requires that
    the model has more than one outputs.
    """

    def __init__(self, net, which_logit):
        super().__init__()
        self.net = net
        self.net.eval()
        self.which_logit = which_logit

    def forward(self, x):
        assert x.shape[1:] == torch.Size([3, 32, 32])
        target_logits = forward(self.net, x, normalization='cifar10',
                                which_logit=self.which_logit)
        if self.which_logit == 'first':
            # l0 = -target_logits
            # l0 = torch.zeros(target_logits.shape, device=x.device) - 50
            l0 = torch.zeros(target_logits.shape, device=x.device)
            # l0 = target_logits.detach().clone()
        else:
            l0 = torch.zeros(target_logits.shape, device=x.device)
            # l0 = target_logits.detach().clone() + 0.1
        return torch.stack([l0, target_logits], dim=1)


# class AutoAttackMaxModel(torch.nn.Module):
#     """
#     A wrapper class to mimic the a two-outputs model for two class
#     classificaiton.
#     This wrapper is required because AutoAttack requires that
#     the model has more than one outputs.
#     """
#     def __init__(self, net):
#         super().__init__()
#         self.net = net
#         self.net.eval()
# 
#     def forward(self, x):
#         assert x.shape[1:] == torch.Size([3, 32, 32])
#         max_logits = forward(self.net, x, normalization='cifar10',
#         which_logit='max')
#         print(max_logits.data)
#         l0 = max_logits.detach().clone()+0.1
#         return torch.stack([l0, max_logits], dim=1)


class AutoAttackModel224(torch.nn.Module):
    """
    A wrapper class to mimic the a two-outputs model for two class
    classificaiton.
    This wrapper is required because AutoAttack requires that
    the model has more than one outputs.
    """

    def __init__(self, net, which_logit):
        super().__init__()
        self.net = net
        self.net.eval()
        self.which_logit = which_logit

    def forward(self, x):
        assert x.shape[1:] == torch.Size([3, 224, 224])
        target_logits = forward(self.net, x, normalization='imagenet',
                                which_logit=self.which_logit)
        if self.which_logit == 'first':
            l0 = torch.zeros(target_logits.shape, device=x.device)
        else:
            l0 = torch.zeros(target_logits.shape, device=x.device)
        return torch.stack([l0, target_logits], dim=1)


def roc_auc(pos_data, neg_data, model, which_logit):
    device = next(model.parameters()).device
    assert not model.training
    normalization = 'cifar10' if pos_data.shape[-1] == 32 else 'imagenet'
    with torch.no_grad():
        # pos_data_out = forward(model, pos_data, normalization)
        pos_data_out = torch.cat(
            [forward(model, batch.to(device), normalization,
                     which_logit=which_logit) for batch in
             torch.split(pos_data, 100)])
        torch.cuda.empty_cache()
        # neg_data_out = forward(model, neg_data, normalization)
        neg_data_out = torch.cat(
            [forward(model, batch.to(device), normalization,
                     which_logit=which_logit) for batch in
             torch.split(neg_data, 100)])
        torch.cuda.empty_cache()
    return auto_auc(pos_data_out, neg_data_out)


def auto_auc(x1, x0):
    x = torch.cat([x1.cpu(), x0.cpu()])
    target = torch.cat([torch.ones(x1.shape[0], dtype=torch.float32),
                        torch.zeros(x0.shape[0], dtype=torch.float32)])
    fpr_, tpr_, thresholds = roc_curve(target, x)
    result = auc_fn(fpr_, tpr_)
    return result


def compute_adv(x, model, attack_config, num_random_restarts=1, sequence=False,
                show_progress=False):
    device = next(model.parameters()).device
    assert not model.training
    assert x.shape[-1] in [32, 128, 256, 224, 512]
    normalization = 'imagenet' if x.shape[-1] != 32 else 'cifar10'

    # If epsilon is 0
    if attack_config['eps'] < 1e-8:
        return x.clone()

    adv = []
    if show_progress:
        from tqdm import tqdm
        data = tqdm(torch.split(x, 100), position=0, leave=True)
    else:
        data = torch.split(x, 100)

    for batch in data:
        if sequence:
            batch_adv = perturb_sequence(model, batch.to(device),
                                         normalization=normalization,
                                         **attack_config)
        else:
            if num_random_restarts > 1:
                batch_adv = perturb_random_restarts(model, batch.to(device),
                                                    normalization=normalization,
                                                    num_random_restarts=num_random_restarts,
                                                    **attack_config)
            else:
                batch_adv = perturb(model, batch.to(device),
                                    normalization=normalization,
                                    **attack_config)
        adv.append(batch_adv)
    adv = torch.cat(adv)
    #     print(torch.norm(ood_x_test.view([200, -1]) - ood_x_test_adv.view([
    #     200, -1]), p=2, dim=1)[:10])
    torch.cuda.empty_cache()
    return adv.cpu()


def ood_adv(model, indist_data, ood_data, epsilon, steps, step_size):
    attack_config = dict(norm='L2', eps=epsilon, steps=steps,
                         step_size=step_size)
    ood_data_adv = compute_adv(ood_data, model, attack_config)
    roc_auc(indist_data, ood_data_adv, model)


def load_dataset(dataset, num_samples=None, size=None, datadir='./datasets'):
    """Return a dataset given the dataset name."""
    assert size in [32, 256]
    transform = transforms.Compose(
        [transforms.Resize([size, size]), transforms.ToTensor()])

    if dataset == 'Gaussian noise':
        assert num_samples is not None
        torch.manual_seed(0)
        gaussian_noise = torch.randn([num_samples, 3, size, size])
        gaussian_noise -= gaussian_noise.min()
        gaussian_noise /= gaussian_noise.max()
        return torch.utils.data.TensorDataset(gaussian_noise)
    elif dataset == 'uniform-noise':
        assert num_samples is not None
        torch.manual_seed(0)
        uniform_noise = torch.rand([num_samples, 3, size, size])
        return torch.utils.data.TensorDataset(uniform_noise)
    elif dataset == 'cifar10':
        return torchvision.datasets.CIFAR10(datadir, train=False,
                                            transform=transform, download=True)
    elif dataset == 'svhn':
        return torchvision.datasets.SVHN(datadir, split='test',
                                         transform=transform, download=True)
    elif dataset == 'cifar100':
        return torchvision.datasets.CIFAR100(datadir, train=False,
                                             transform=transform, download=True)
    elif dataset == 'imagenet32':
        assert size == 32
        return get_imagenet32_val_dataset(datadir)
    elif dataset == 'TinyImages':
        assert size == 32
        return TinyImages(datafile=os.path.join(datadir, 'tinyimages1000k.npy'),
                          transform=transform)
    elif dataset == 'afhqcat256':
        return get_afhq256_dataset(datadir, subset='cat')
    elif dataset == 'celebahq256':
        return get_celebahq256_dataset(datadir)
    elif dataset == 'church256':
        return get_church256_dataset(datadir)
    elif dataset == 'imagenetval':
        return get_imagenet256_val_dataset(datadir)
    else:
        raise ValueError('Dataset not supported')


def load_data(dataset, test_samples):
    if test_samples == -1:
        # Load all the data
        loader = torch.utils.data.DataLoader(dataset, batch_size=100,
                                             shuffle=True)
        all_data = []
        for data in loader:
            if isinstance(data, list):
                data = data[0]
            all_data.append(data)
        return torch.cat(all_data)
    else:
        torch.manual_seed(0)  # Make sure to set the seed
        loader = torch.utils.data.DataLoader(dataset, batch_size=test_samples,
                                             shuffle=True)
        # Only want the data (not the labels if there are any)
        data = iter(loader).next()
        if isinstance(data, list):
            data = data[0]
        return data


def eval_ood_detection_clean(model, indist_dataset, ood_dataset, size,
                                 eval_samples, which_logit='first',
                                 indist_samples=-1):
    """
    Evaluate standard out-of-distribution detection performance
    :param model: first logit output of the model indicates input's
    likelihood to be in-distribution data
    :param indist_dataset
    :param ood_dataset
    :param eval_samples: number of samples used for this evaluation
    :return: AUC score on in-distribution data and OOD data
    """
    assert size in [32, 256]
    data_shape = torch.Size([3, size, size])

    indist_data = load_data(indist_dataset, indist_samples)
    ood_data = load_data(ood_dataset, eval_samples)
    assert indist_data.shape[1:] == data_shape
    assert ood_data.shape[1:] == data_shape

    with torch.no_grad():
        auc_score = roc_auc(indist_data, ood_data, model, which_logit)
        return auc_score


def eval_ood_detection_autoattack(model, indist_dataset, ood_dataset, size,
                                  eval_samples,
                                  which_logit='first', n_restarts=5,
                                  indist_samples=-1):
    """
    Evaluate adversarial out-of-distribution detection performance; use
    AutoAttack to perturb OOD data
    :param model: first logit output of the model indicates input's
    likelihood to be in-distribution data
    :param indist_dataset
    :param ood_dataset
    :param eval_samples: number of samples used for this evaluation
    :return: AUC score on in-distribution data and perturbed OOD data
    """
    assert which_logit in ['first', 'max']
    assert size in [32, 256]
    data_shape = torch.Size([3, size, size])

    indist_data = load_data(indist_dataset, indist_samples)
    ood_data = load_data(ood_dataset, eval_samples)
    ood_data_labels = torch.zeros(ood_data.shape[0], dtype=torch.int64)
    assert indist_data.shape[1:] == data_shape
    assert ood_data.shape[1:] == data_shape

    if size == 32:
        modelW = Cifar10Wrapper(model, which_logit)
        adversary = AutoAttack(modelW, norm='L2', eps=1.0, verbose=False)
    elif size == 256:
        modelW = ImageNetWrapper(model, which_logit)
        adversary = AutoAttack(modelW, norm='L2', eps=7.0, verbose=False)

    adversary.attacks_to_run = ['apgd-ce']
    adversary.apgd.loss = {'first': 'first_logit', 'max': 'max_logit'}[which_logit]
    adversary.apgd.n_restarts = n_restarts
    adversary.apgd.n_iter = 100

    # adversary.attacks_to_run = ['fab']
    # adversary.fab.n_restarts = 1
    # adversary.fab.n_iter = 100

    # adversary.attacks_to_run = ['square']
    # adversary.square.n_queries = 5000
    with torch.no_grad():
        bs = {32: 200, 256: 100}[size]
        ood_data_adv = adversary.run_standard_evaluation(ood_data,
                                                         ood_data_labels, bs=bs)
        auc_score = roc_auc(indist_data, ood_data_adv, model, which_logit)
        return auc_score


class NormalizationWrapper(torch.nn.Module):
    def __init__(self, model, mean, std):
        super().__init__()

        mean = mean[..., None, None]
        std = std[..., None, None]

        self.train(model.training)

        self.model = model
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(self, x, *args, **kwargs):
        x_normalized = (x - self.mean) / self.std
        return self.model(x_normalized, *args, **kwargs)

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        return self.model.state_dict()


def Cifar10Wrapper(model, which_logit):
    assert which_logit in ['first', 'max']
    if which_logit == 'first':
        mean = torch.tensor([0.4914, 0.4822, 0.4465]).to('cuda:0')
        std = torch.tensor([0.2023, 0.1994, 0.2010]).to('cuda:0')
    else:
        mean = torch.tensor(
            [0.4913997551666284, 0.48215855929893703, 0.4465309133731618]).to(
            'cuda:0')
        std = torch.tensor(
            [0.24703225141799082, 0.24348516474564, 0.26158783926049628]).to(
            'cuda:0')
    return NormalizationWrapper(model, mean, std)


def ImageNetWrapper(model, which_logit):
    assert which_logit in ['first', 'max']
    if which_logit == 'first':
        mean = torch.tensor([0.485, 0.456, 0.406]).to('cuda:0')
        std = torch.tensor([0.229, 0.224, 0.225]).to('cuda:0')
    else:
        mean = torch.tensor([0.4717, 0.4499, 0.3837]).to('cuda:0')
        std = torch.tensor([0.2600, 0.2516, 0.2575]).to('cuda:0')
    return NormalizationWrapper(model, mean, std)


def generate(datasize, samples, savedir, attack_config, model, batch_size=None,
             seed=0):
    assert datasize in [32, 256]
    assert not model.training
    print(attack_config)

    # Load seed images
    if datasize == 32:
        ood_dataset = load_dataset('TinyImages')
    else:
        ood_dataset = get_imagenet256_dataset(datadir='./datasets')

    if batch_size is None:
        batch_size = {32: 5000, 256: 100}[datasize]

    assert samples % batch_size == 0
    max_iters = samples // batch_size

    if datasize == 32:
        ood_samples = torch.load('./data/cifar_fid_imgs.pt')
    else:
        ood_samples = torch.load('./data/imagenet_fid_10K_samples.pt')
    print('loaded ood_samples')
    ood_dataset = torch.utils.data.TensorDataset(ood_samples)
    loader = torch.utils.data.DataLoader(ood_dataset, batch_size=batch_size,
                                         shuffle=False)

    for i, data in tqdm(zip(range(max_iters), loader), total=max_iters):
        # Discard labels
        # print('i', end='-')
        seed_imgs = data[0] if isinstance(data, list) else data
        # print('v', end='-')
        adv = compute_adv(seed_imgs, model, attack_config)

        # print('s', end='-')
        # Save generated images
        for j in range(adv.shape[0]):
            transforms.ToPILImage()(adv[j]).save(
                os.path.join(savedir, f'{i * batch_size + j:06d}.png'))


def compute_fid(dataset, model, savedir):
    Path(savedir).mkdir(parents=True, exist_ok=True)
    attack_configs = {
        'cifar10': dict(norm='L2', eps=10000, steps=32, step_size=0.2),
        'celebahq256': dict(norm='L2', eps=10000, steps=10, step_size=16.0),
        'afhq256': dict(norm='L2', eps=10000, steps=7, step_size=16.0),
        'church256': dict(norm='L2', eps=10000, steps=10, step_size=16.0),
    }
    gt_dirs = {
        'cifar10': 'data/fid_stats_cifar10_train.npz',
        'celebahq256': './datasets/CelebAHQ256/train/png',
        'afhq256': './datasets/AFHQ-png/afhq256/train/cat/data',
        'church256': './datasets/Church256/train/data',
    }
    samples = {
        'cifar10': 10000,
        'celebahq256': 10000,
        'afhq256': 10000,
        'church256': 10000,
    }
    datasize = 32 if dataset == 'cifar10' else 256
    assert dataset in attack_configs.keys()
    generate(datasize=datasize, samples=samples[dataset], savedir=savedir,
             attack_config=attack_configs[dataset], model=model)
    assert len([name for name in os.listdir(savedir) if
                os.path.isfile(os.path.join(savedir, name))]) == samples[
               dataset]
    return calculate_fid_given_paths([savedir, gt_dirs[dataset]],
                                     batch_size=200,
                                     device=torch.device('cuda'), dims=2048,
                                     num_workers=4)


