import argparse
import matplotlib.pyplot as plt
from eval_utils import load_dataset, compute_adv
from utils import set_eval
from models import resnet50
import pathlib
from torchvision.utils import make_grid
import torchvision
from PIL import Image
import torchvision.models as models
import torchvision.transforms as transforms
from starganv2.core.model import Discriminator
from dataset import get_imagenet256_dataset
from tqdm import tqdm
import torch.nn as nn
import torch
import os

datadir = './datasets'
attack_configs = {
    'cifar10': dict(norm='L2', eps=10000, steps=32, step_size=0.2),
    'celebahq256': dict(norm='L2', eps=10000, steps=20, step_size=8.0),
    'afhqcat256': dict(norm='L2', eps=10000, steps=14, step_size=8.0),
    'church256': dict(norm='L2', eps=10000, steps=17, step_size=8.0)
}
model_files = {
    'cifar10': 'experiments/cifar10-stepsize0.1-lr0.0005-optimizersgd-E5'
               '-generation-bs32-r1reg0.01/model.pth',
    'celebahq256': 'experiments/celebahq256-stepsize2.0-lr5e-05-optimizeradam'
                   '-E5-celebahq256-stargan-wd0-r1reg30-th99-bestfid/model.pth',
    'afhqcat256': 'experiments/afhq256-stepsize2.0-lr5e-05-optimizeradam-E50'
                  '-afhq256-startsteps10-adamfixed-pretrain-stargan-cat'
                  '-r1reg100/model.pth',
    'church256': 'experiments/church256-stepsize2.0-lr5e-05-optimizeradam-E1'
                 '-stargan-wd0-r1reg100-th90/model.pth'
}

ToPILImage = transforms.ToPILImage()
ToTensor = transforms.ToTensor()

device = 'cuda:0'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str,
                        choices=['cifar10', 'celebahq256', 'afhqcat256', 'church256'], 
                        required=True)
    parser.add_argument('--samples', type=int, default=50000)
    parser.add_argument('--savedir', type=str, required=True)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--model', type=str, default=None)
    args = parser.parse_args()

    # Load model
    model = resnet50() if args.dataset == 'cifar10' else Discriminator(num_classes=1000)
    model = nn.DataParallel(model)
    model = model.to(device)
    if args.model:
        model.load_state_dict(torch.load(args.model))
    else:
        model.load_state_dict(torch.load(model_files[args.dataset]))
    set_eval(model)

    # Load seed images
    if args.dataset == 'cifar10':
        ood_dataset = load_dataset('TinyImages')
    else:
        ood_dataset = get_imagenet256_dataset(datadir='./datasets')

    batch_size = 5000 if args.dataset == 'cifar10' else 100
    if args.batch_size is not None:
        batch_size = args.batch_size
    assert args.samples % batch_size == 0
    max_iters = args.samples // batch_size
    pathlib.Path(args.savedir).mkdir(parents=True, exist_ok=True)
    # Make reproducible
    if args.seed is not None:
        torch.manual_seed(args.seed)
    else:
        torch.manual_seed(0)
    loader = torch.utils.data.DataLoader(ood_dataset, batch_size=batch_size,
                                         shuffle=True)
    attack_config = attack_configs[args.dataset]
    print(attack_config)
    for i, data in tqdm(zip(range(max_iters), loader), total=max_iters):
        # Discard labels
        seed_imgs = data[0] if isinstance(data, list) else data
        adv = compute_adv(seed_imgs, model, attack_config)

        # Save generated images
        for j in range(adv.shape[0]):
            img = transforms.ToPILImage()(adv[j])
            img.save(os.path.join(args.savedir, f'{i * batch_size + j:06d}.png'))

