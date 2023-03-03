import os
import sys
import torch
import torchvision
import torchvision.transforms as transforms
from dataset import *
from pgd_attack import perturb
from sklearn.metrics import roc_curve, auc as compute_auc
from tqdm import tqdm
import pathlib

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
            # https://github.com/M4xim4l/InNOutRobustness/blob
            # /c649f1f94d84e5a4ea1abf9636496f6a171e0c79/utils
            # /model_normalization.py#L29
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
            # https://github.com/M4xim4l/InNOutRobustness/blob
            # /c649f1f94d84e5a4ea1abf9636496f6a171e0c79/utils
            # /model_normalization.py#L58
            mean = torch.as_tensor([0.4717, 0.4499, 0.3837], dtype=x.dtype,
                                   device=x.device)
            std = torch.as_tensor([0.2600, 0.2516, 0.2575], dtype=x.dtype,
                                  device=x.device)
    logits = model((x - mean.view(1, 3, 1, 1)) / std.view(1, 3, 1, 1))
    return {'all': logits, 'first': logits[:, 0],
            'max': torch.max(logits, dim=1)[0]}[which_logit]


def compute_auroc(indist_data, outdist_data, model, which_logit):
    """Compute AUROC score."""
    device = next(model.parameters()).device
    assert not model.training
    normalization = 'cifar10' if indist_data.shape[-1] == 32 else 'imagenet'
    with torch.no_grad():
        indist_pred = torch.cat(
            [forward(model, batch.to(device), normalization, which_logit) for
             batch in torch.split(indist_data, 100)])
        outdist_pred = torch.cat(
            [forward(model, batch.to(device), normalization, which_logit) for
             batch in torch.split(outdist_data, 100)])
        torch.cuda.empty_cache()
    pred = torch.cat([indist_pred.cpu(), outdist_pred.cpu()])
    target = torch.cat([torch.ones(indist_pred.shape[0], dtype=torch.float32),
                        torch.zeros(outdist_pred.shape[0],dtype=torch.float32)])
    fpr_, tpr_, thresholds = roc_curve(target, pred)
    return compute_auc(fpr_, tpr_)



def load_dataset(dataset, num_samples=None, size=None, datadir='./datasets'):
    """Load dataset based on the dataset name."""
    if size is not None:
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
    elif dataset == 'afhqcat256':
        return get_afhq256_dataset(datadir, subset='cat')
    elif dataset == 'celebahq256':
        return get_celebahq256_dataset(datadir)
    elif dataset == 'church256':
        return get_church256_dataset(datadir)
    elif dataset == 'imagenetval':
        return get_imagenet256_val_dataset(datadir)
    else:
        raise ValueError('Unkown Dataset')


def load_data(dataset, samples):
    """Load samples from a dataset."""
    if samples == -1:
        # Load all the data
        loader = torch.utils.data.DataLoader(dataset, batch_size=100,
                                             shuffle=True)
        all_data = []
        for data in loader:
        # Discard labels
            if isinstance(data, list):
                data = data[0]
            all_data.append(data)
        return torch.cat(all_data)
    else:
        torch.manual_seed(0)
        loader = torch.utils.data.DataLoader(dataset, batch_size=samples,
                                             shuffle=True)
        data = iter(loader).next()
        # Discard labels
        if isinstance(data, list):
            data = data[0]
        return data


def eval_ood_detection_clean(model, indist_dataset, ood_dataset, size,
                                 eval_samples, which_logit='first'):
    """
    Evaluate standard out-of-distribution detection
    :param model: first logit output of the model indicates input's
    likelihood to be in-distribution data
    :param indist_dataset
    :param ood_dataset
    :param size: image size
    :param eval_samples: number of samples for this evaluation
    :return: AUC score
    """
    assert size in [32, 256]
    data_shape = torch.Size([3, size, size])

    indist_data = load_data(indist_dataset, eval_samples)
    ood_data = load_data(ood_dataset, eval_samples)
    assert indist_data.shape[1:] == data_shape
    assert ood_data.shape[1:] == data_shape

    with torch.no_grad():
        auc_score = compute_auroc(indist_data, ood_data, model, which_logit)
        return auc_score


def eval_ood_detection_autoattack(model, indist_dataset, ood_dataset, size,
                                  eval_samples, which_logit='first',
                                  n_restarts=5, batch_size=100):
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

    indist_data = load_data(indist_dataset, eval_samples)
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
    adversary.apgd.loss = {'first': 'first_logit', 
                           'max': 'max_logit'}[which_logit]
    adversary.apgd.n_restarts = n_restarts
    adversary.apgd.n_iter = 100

    # adversary.attacks_to_run = ['fab']
    # adversary.fab.n_restarts = 1
    # adversary.fab.n_iter = 100

    # adversary.attacks_to_run = ['square']
    # adversary.square.n_queries = 5000
    with torch.no_grad():
        ood_data_adv = adversary.run_standard_evaluation(ood_data,
                                                         ood_data_labels,
                                                         bs=batch_size)
        auc_score = compute_auroc(indist_data, ood_data_adv, model, which_logit)
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
    device = next(model.parameters()).device
    assert which_logit in ['first', 'max']
    if which_logit == 'first':
        mean = torch.tensor([0.4914, 0.4822, 0.4465]).to(device)
        std = torch.tensor([0.2023, 0.1994, 0.2010]).to(device)
    else:
        # https://github.com/M4xim4l/InNOutRobustness/blob
        # /c649f1f94d84e5a4ea1abf9636496f6a171e0c79/utils/model_normalization
        # .py#L29
        mean = torch.tensor(
            [0.4913997551666284, 0.48215855929893703, 0.4465309133731618])
        std = torch.tensor(
            [0.24703225141799082, 0.24348516474564, 0.26158783926049628])
        mean, std = mean.to(device), std.to(device)
    return NormalizationWrapper(model, mean, std)


def ImageNetWrapper(model, which_logit):
    device = next(model.parameters()).device
    assert which_logit in ['first', 'max']
    if which_logit == 'first':
        mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
        std = torch.tensor([0.229, 0.224, 0.225]).to(device)
    else:
        mean = torch.tensor([0.4717, 0.4499, 0.3837]).to(device)
        std = torch.tensor([0.2600, 0.2516, 0.2575]).to(device)
    return NormalizationWrapper(model, mean, std)


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
    torch.cuda.empty_cache()
    return adv.cpu()


def generate(datasize, samples, savedir, attack_config, model, batch_size=None):
    assert datasize in [32, 256]
    assert not model.training
    print(attack_config)

    if batch_size is None:
        batch_size = {32: 5000, 256: 100}[datasize]

    assert samples % batch_size == 0
    max_iters = samples // batch_size

    if datasize == 32:
        ood_samples = torch.load('./data/cifar_fid_imgs_50K.pt')[:samples]
        ood_dataset = torch.utils.data.TensorDataset(ood_samples)
        loader = torch.utils.data.DataLoader(ood_dataset, batch_size=batch_size,
                                             shuffle=False)
    else:
        # ood_dataset = torchvision.datasets.ImageFolder(os.path.join(datadir, 'imagenet50K'), transform=ToTensor())
        ood_dataset = get_imagenet256_dataset(datadir='./datasets')
        # https://pytorch.org/docs/stable/data.html#torch.utils.data.random_split
        loader = torch.utils.data.DataLoader(ood_dataset, batch_size=batch_size,
                                             shuffle=True, generator=torch.Generator().manual_seed(0))
    print('loaded ood_samples')

    pathlib.Path(savedir).mkdir(parents=True, exist_ok=True)
    for i, data in tqdm(zip(range(max_iters), loader), total=max_iters):
        seed_imgs = data[0] if isinstance(data, list) else data
        adv = compute_adv(seed_imgs, model, attack_config)

        # Save generated images
        for j in range(adv.shape[0]):
            transforms.ToPILImage()(adv[j]).save(
                os.path.join(savedir, f'{i * batch_size + j:06d}.png'))


def compute_fid(dataset, model, savedir):
    pathlib.Path(savedir).mkdir(parents=True, exist_ok=True)
    attack_configs = {
        'cifar10': dict(norm='L2', eps=10000, steps=32, step_size=0.2),
        'celebahq256': dict(norm='L2', eps=10000, steps=10, step_size=16.0),
        'afhq256': dict(norm='L2', eps=10000, steps=7, step_size=16.0),
        'church256': dict(norm='L2', eps=10000, steps=10, step_size=16.0),
    }
    gt_dirs = {
        'cifar10': 'data/fid_stats_cifar10_train.npz',
        'celebahq256': './datasets/CelebAHQ256/train/data',
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


