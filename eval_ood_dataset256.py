import argparse
import torch
import numpy as np
import torch.nn as nn
from dataset import *
from misc import set_eval
from eval_utils import load_dataset256
from eval_utils import eval_ood_detection_autoattack, \
    eval_ood_detection_noperturb
from starganv2.core.model import Discriminator

parser = argparse.ArgumentParser()
parser.add_argument('--task', type=str,
                    choices=['celebahq256', 'afhqcat256', 'church256'],
                    required=True)
parser.add_argument('--restarts', type=int, default=5)
parser.add_argument('--eval_samples', type=int, default=1024)
args = parser.parse_args()

checkpoint = {
    'celebahq256': 'experiments/celebahq256-stepsize2.0-lr5e-05-optimizeradam'
                   '-E5-celebahq256-stargan-wd0-r1reg30-th99-bestfid/model.pth',
    'afhqcat256': 'experiments/afhq256-stepsize2.0-lr5e-05-optimizeradam-E50'
                  '-afhq256-startsteps10-adamfixed-pretrain-stargan-cat'
                  '-r1reg100/model.pth',
    'church256': 'experiments/church256-stepsize2.0-lr5e-05-optimizeradam-E1'
                 '-stargan-wd0-r1reg100-th90/model.pth'}

device = 'cuda:0'
datadir = './datasets'


def run_ood_eval(model, task):
    ood_dataset_names = ['CIFAR10', 'Uniform noise', 'ImageNetVal', 'SVHN']
    if task == 'celebahq256':
        ood_dataset_names.extend(['afhq-cat', 'church'])
        indist_dataset = get_celebahq256_dataset(datadir='./datasets')
    elif task == 'afhqcat256':
        ood_dataset_names.extend(['celebahq', 'church'])
        indist_dataset = get_afhq256_dataset(datadir='./datasets', subset='cat')
    elif task == 'church256':
        ood_dataset_names.extend(['celebahq', 'afhq-cat'])
        indist_dataset = get_church256_dataset(datadir='./datasets')

    for ood_dataset_name in ood_dataset_names:
        print(ood_dataset_name)
        ood_dataset = load_dataset256(ood_dataset_name,
                                      num_samples=args.eval_samples)
        clean_aug = eval_ood_detection_noperturb(model, indist_dataset,
                                                 ood_dataset,
                                                 data_size=256,
                                                 eval_samples=args.eval_samples,
                                                 which_logit='first',
                                                 indist_samples=args.eval_samples)
        adv_auc = eval_ood_detection_autoattack(model, indist_dataset,
                                                ood_dataset, data_size=256,
                                                eval_samples=args.eval_samples,
                                                which_logit='first',
                                                n_restarts=args.restarts,
                                                verbose=True,
                                                indist_samples=args.eval_samples)
        print(f'{task} clean / adv auc: {clean_aug: .4f} / {adv_auc: .4f}')


model = Discriminator(num_classes=1000)
model = nn.DataParallel(model)
model = model.to(device)
model.load_state_dict(torch.load(checkpoint[args.task]))
set_eval(model)
run_ood_eval(model, args.task)

