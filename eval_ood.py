import argparse
import torch
import numpy as np
import torch.nn as nn
from misc import set_eval
from eval_utils import load_dataset
from eval_utils import eval_ood_detection_autoattack
from eval_utils import eval_ood_detection_clean
from models import resnet50
from starganv2.core.model import Discriminator

parser = argparse.ArgumentParser()
parser.add_argument('--task', type=str,
                    choices=['cifar10', 'celebahq256', 'afhqcat256', 'church256'],
                    required=True)
parser.add_argument('--restarts', type=int, default=5)
parser.add_argument('--eval_samples', type=int, default=1024)
parser.add_argument('--batch_size', type=int, default=100)
args = parser.parse_args()

checkpoint = {
    'cifar10': 'experiments/cifar10-stepsize0.1-lr0.1-optimizersgd-E100-ood'
               '-replicate-ratio-nodecay-noBN-advonly-nopretrain-eps0.5'
               '-ineps0.25/model.pth',
    'celebahq256': 'experiments/celebahq256-stepsize2.0-lr5e-05-optimizeradam'
                   '-E5-celebahq256-stargan-wd0-r1reg30-th99-bestfid/model.pth',
    'afhqcat256': 'experiments/afhq256-stepsize2.0-lr5e-05-optimizeradam-E50'
                  '-afhq256-startsteps10-adamfixed-pretrain-stargan-cat'
                  '-r1reg100/model.pth',
    'church256': 'experiments/church256-stepsize2.0-lr5e-05-optimizeradam-E1'
                 '-stargan-wd0-r1reg100-th90/model.pth'}

device = 'cuda:0'
datadir = './datasets'
np.random.seed(123)
torch.manual_seed(123)


def eval_ood(model, task):
    image_size = 32 if task == 'cifar10' else 256
    indist_dataset = load_dataset(task, size=image_size)
    ood_datasets = {
        'cifar10': ['svhn', 'cifar100', 'imagenet32', 'uniform-noise'],
        'celebahq256': ['afhqcat256', 'church256', 'cifar10', 'uniform-noise',
                        'imagenetval', 'svhn'],
        'afhqcat256': ['celebahq256', 'church256', 'cifar10', 'uniform-noise',
                       'imagenetval', 'svhn'],
        'church256': ['celebahq256', 'afhqcat256', 'cifar10', 'uniform-noise',
                      'imagenetval', 'svhn']}[task]

    for ood_dataset in ood_datasets:
        dataset = load_dataset(ood_dataset, num_samples=args.eval_samples,
                               size=image_size)
        clean_auc = eval_ood_detection_clean(model, indist_dataset,
                                             dataset,
                                             size=image_size,
                                             eval_samples=args.eval_samples,
                                             which_logit='first')
        adv_auc = eval_ood_detection_autoattack(model, indist_dataset,
                                                dataset, size=image_size,
                                                eval_samples=args.eval_samples,
                                                which_logit='first',
                                                n_restarts=args.restarts,
                                                batch_size=args.batch_size)
        print(f'{task}/{ood_dataset} clean auc / adv auc: {clean_auc: .4f} / {adv_auc: .4f}')


model = resnet50() if args.task == 'cifar10' else Discriminator(num_classes=1000)
model = nn.DataParallel(model)
model = model.to(device)
model.load_state_dict(torch.load(checkpoint[args.task]))
set_eval(model)
eval_ood(model, args.task)


