import torch
import argparse
from models import resnet50
import numpy as np
import torch.nn as nn
from dataset import *
from misc import set_eval
from eval_utils import load_dataset
from eval_utils import eval_ood_detection_autoattack
from eval_utils import eval_ood_detection_noperturb

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str,
                    default='experiments/cifar10-stepsize0.1-lr0.1'
                            '-optimizersgd-E100-ood-replicate-ratio-nodecay'
                            '-noBN-advonly-nopretrain-eps0.5-ineps0.25/model'
                            '.pth')
parser.add_argument('--dataset', type=str,
                    choices=['SVHN', 'CIFAR100', 'imagenet32', 'Uniform noise'],
                    default='CIFAR100')
parser.add_argument('--samples', type=int, default=1024)
parser.add_argument('--restarts', type=int, default=5)
args = parser.parse_args()

device = 'cuda:0'
datadir = './datasets'
np.random.seed(123)
torch.manual_seed(123)

indist_dataset = load_dataset('CIFAR10')
ood_dataset = load_dataset(args.dataset, data_size=32, num_samples=args.samples)

model = resnet50()
model = nn.DataParallel(model)
model = model.to(device)

model.load_state_dict(torch.load(args.model))
set_eval(model)

nat_auc = eval_ood_detection_noperturb(model, indist_dataset, ood_dataset,
                                       eval_samples=args.samples,
                                       which_logit='first', data_size=32)
adv_auc = eval_ood_detection_autoattack(model, indist_dataset, ood_dataset,
                                        eval_samples=args.samples,
                                        which_logit='first', data_size=32,
                                        verbose=False, n_restarts=args.restarts)
print(f'nat auc {nat_auc}, adv_auc {adv_auc}')

