import argparse
from misc import set_eval
from models import resnet50
import torchvision.transforms as transforms
from starganv2.core.model import Discriminator
from eval_utils import generate
import torch.nn as nn
import torch

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
                        choices=['cifar10', 'celebahq256', 'afhqcat256',
                                 'church256'], required=True)
    parser.add_argument('--samples', type=int, default=50000)
    parser.add_argument('--savedir', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--model', type=str, default=None)
    args = parser.parse_args()

    # Load model
    model = resnet50() if args.dataset == 'cifar10' else \
        Discriminator(num_classes=1000)
    model = nn.DataParallel(model)
    model = model.to(device)
    if args.model:
        model.load_state_dict(torch.load(args.model))
    else:
        model.load_state_dict(torch.load(model_files[args.dataset]))
    set_eval(model)

    attack_config = attack_configs[args.dataset]
    datasize = 32 if args.dataset == 'cifar10' else 256
    generate(datasize, args.samples, args.savedir, attack_config, model, args.batch_size)


