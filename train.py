import argparse
import os
import sys
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets
from sklearn.metrics import roc_curve, auc as auc_fn
from torch.utils.tensorboard import SummaryWriter
import random
from functools import partial
import math

from models import resnet50
from starganv2.core.model import Discriminator
from dataset import get_celebahq256_dataset
from dataset import get_church256_dataset
from dataset import get_imagenet256_dataset
from dataset import get_afhq256_dataset
from pgd_attack import perturb, forward
from misc import save_model
from misc import set_train
from GOOD.tiny_utils.tinyimages_80mn_loader import TinyImages
from collections import deque
from misc import r1_reg
from eval_utils import compute_fid

sys.path.append('./InNOutRobustness')
import InNOutRobustness.utils.datasets as dl

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str,
                    choices=['cifar10', 'celebahq256', 'afhq256-cat', 'church256'],
                    required=True)
parser.add_argument('--eps', type=float, default=math.inf,
                    help='Perturbation limit for out-distribution adversarial attack')
parser.add_argument('--max_steps', type=int, required=True,
                    help='Number of PGD steps for out-distribution adversarial attack')
parser.add_argument('--startstep', type=int, default=0)
parser.add_argument('--step_size', type=float, required=True,
                    help='PGD attack step size')
parser.add_argument('--indist_steps', type=int, default=0,
                    help='Number of PGD steps for in-distribution adversarial attack')
parser.add_argument('--indist_eps', type=float, default=math.inf,
                    help='Perturbation limit for in-distribution adversarial attack')
parser.add_argument('--indist_aug', action='store_true',
                    help='Perform data augmentation on in-distribution data')

parser.add_argument('--optimizer', type=str, default='adam')
parser.add_argument('--wd', type=float, default=0,
                    help='Weight decay for optimizer')
parser.add_argument('--lr', type=float, required=True,
                    help='Learning rate')
parser.add_argument('--r1reg', type=float, default=0,
                    help='Weight for R1 regularization')
parser.add_argument('--pretrain', action='store_true',
                    help='Use pretrained model for the D model')
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--epochs_per_step', type=int, required=True,
                    help='number of training epochs for each step')
parser.add_argument('--AUC_th', type=float, default=0.8,
                    help='When the AUC reaches the threshold, interrupt the training')
parser.add_argument('--cifar10_ood_detection', action='store_true')

parser.add_argument('--datadir', type=str, default='./datasets',
                    help='The location of the datasets')
parser.add_argument('--logdir', type=str, default='./runs',
                    help='The location to store models and tensorboard logs')
parser.add_argument('--logfid', action='store_true')
parser.add_argument('--fid_log_interval', type=int, default=500,
                    help='Log fid every 500 iterations')

parser.add_argument('--rand_seed', type=int, default=0)
parser.add_argument('--resume', action='store_true',
                    help='Resume model and optimizer checkpoints')
parser.add_argument('--comment', type=str, default='',
                    help='Comment to be added to the task signature')

args = parser.parse_args()

def get_task_signature(args):
    exclude = ['resume', 'logfid', 'fid_log_interval', 'datadir', 'logdir']
    if args.dataset != 'cifar10':
        exclude.append('cifar10_ood_detection')
    if args.dataset == 'cifar10':
        exclude.extend(['indist_aug', 'pretrain'])
    sig = []
    for k, v in vars(args).items():
        if str(k) not in exclude:
            key = str(k).replace('_','')
            sig.append(f'{key}{v}')
    return '-'.join(sig)


def log_fid():
    model.eval()
    savedir = os.path.join('eval_fid', taskdir)
    print('computing fid ... ', end=' ')
    fid = compute_fid(dataset=args.dataset, model=model, savedir=savedir)
    print(f'fid: {fid}')
    tb_logger.add_scalar('fid', fid, global_step)
    return fid


np.random.seed(args.rand_seed)
torch.manual_seed(args.rand_seed)
if args.resume:
    seed = int(datetime.now().timestamp())
    print(f'using rand seed {seed}')
    np.random.seed(seed)
    torch.manual_seed(seed)

# torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
taskdir = os.path.join(args.logdir, get_task_signature(args))
tb_logger = SummaryWriter(taskdir)
global_step = 0
normalization = 'cifar10' if args.dataset in ['cifar10'] else 'imagenet'

# Setup data laoder
if args.dataset in ['cifar10']:
    if args.cifar10_ood_detection:
        indist_loader = dl.get_CIFAR10(train=True, batch_size=args.batch_size,
                                       augm_type='autoaugment_cutout',
                                       size=32, config_dict={})
        outdist_loader = dl.get_80MTinyImages(batch_size=args.batch_size,
                                              augm_type='autoaugment_cutout',
                                              num_workers=1, size=32,
                                              exclude_cifar=True,
                                              exclude_cifar10_1=True,
                                              config_dict={})
    else:
        cifar10_transform = transforms.Compose(
            [transforms.RandomCrop(32, padding=4),
             transforms.RandomHorizontalFlip(),
             transforms.ToTensor()])
        outdist_dataset = TinyImages(transform=cifar10_transform,
                                     exclude_cifar=['H', 'CEDA11'])
        outdist_loader = torch.utils.data.DataLoader(outdist_dataset,
                                                     batch_size=args.batch_size,
                                                     shuffle=True,
                                                     num_workers=4,
                                                     pin_memory=True,
                                                     drop_last=True)
        indist_dataset = torchvision.datasets.CIFAR10(root=args.datadir,
                                                      train=True,
                                                      download=True,
                                                      transform=cifar10_transform)
        indist_loader = torch.utils.data.DataLoader(indist_dataset,
                                                    batch_size=args.batch_size,
                                                    shuffle=True, num_workers=4,
                                                    pin_memory=True,
                                                    drop_last=True)

    outdist_loader_iter = iter(outdist_loader)
else:
    if args.indist_aug:
        img_size = 256
        if args.dataset in ['afhq256-cat', 'church256']:
            scale, ratio = (0.8, 1.0), (0.9, 1.1)
        else:
            scale, ratio = (0.9, 1.0), (0.95, 1.05)
        crop = transforms.RandomResizedCrop(img_size, scale=scale, ratio=ratio)
        rand_crop = transforms.Lambda(
            lambda x: crop(x) if random.random() < 0.5 else x)
        indist_transform = transforms.Compose([
            rand_crop,
            transforms.Resize([img_size, img_size]),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()])
    else:
        indist_transform = transforms.Compose(
            [transforms.RandomHorizontalFlip(), transforms.ToTensor()])

    datasets = {'celebahq256': get_celebahq256_dataset,
                'church256': get_church256_dataset,
                'afhq256-cat': partial(get_afhq256_dataset, subset='cat')}
    indist_dataset = datasets[args.dataset](args.datadir, indist_transform)
    indist_loader = torch.utils.data.DataLoader(
        indist_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=8, pin_memory=True, drop_last=True)

    outdist_dataset = get_imagenet256_dataset(args.datadir)
    outdist_loader = torch.utils.data.DataLoader(
        outdist_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=8, pin_memory=True, drop_last=True)

    outdist_loader_iter = iter(outdist_loader)

    print(indist_transform)
    print(outdist_dataset.transform)

# D model
if args.dataset in ['cifar10']:
    model = resnet50()
else:
    model = Discriminator(num_classes=1000)
    if args.pretrain:
        # The pretrained model is trained on the ImgeNet classification task
        # Using the pretrained model gives slightly better FID
        model.load_state_dict(torch.load('checkpoints/stargan/ckpt.pth'))
        print('loaded pretrained model')

model = nn.DataParallel(model)
model.to(device)

criterion = nn.BCEWithLogitsLoss(reduction='mean')
if args.optimizer == 'sgd':
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9,
                                nesterov=True, weight_decay=args.wd)
else:
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 betas=(0.0, 0.99),
                                 weight_decay=args.wd)
print(optimizer)

if args.resume:
    model.load_state_dict(torch.load(os.path.join(taskdir, 'model.pth')))
    optimizer.load_state_dict(torch.load(os.path.join(taskdir, 'optimizer.pth')))
    print('loaded from ' + taskdir)

best_fid = math.inf
for step in range(args.startstep, args.max_steps + 1):
    rolling_adv_auc = deque(maxlen=100)
    step_interrupt = False
    indist_attack_config = dict(norm='L2', eps=args.indist_eps,
                                steps=args.indist_steps,
                                step_size=args.step_size)
    outdist_attack_config = dict(norm='L2', eps=args.eps, steps=step,
                                 step_size=args.step_size)

    max_epochs = args.epochs_per_step if step < args.max_steps else sys.maxsize
    for epoch in range(0, max_epochs):
        tb_logger.flush()

        for i, indist_imgs in enumerate(indist_loader):
            if args.logfid and global_step % args.fid_log_interval == 0:
                fid = log_fid()
                if fid < best_fid:
                    best_fid = fid
                    save_model(model, os.path.join(taskdir, 'model_bestfid.pth'))

            try:
                outdist_imgs = next(outdist_loader_iter)
            except StopIteration:
                outdist_loader_iter = iter(outdist_loader)
                outdist_imgs = next(outdist_loader_iter)

            global_step += 1

            # Discard labels
            if isinstance(indist_imgs, list):
                indist_imgs = indist_imgs[0]
            if isinstance(outdist_imgs, list):
                outdist_imgs = outdist_imgs[0]
            indist_imgs, outdist_imgs = indist_imgs.to(device), outdist_imgs.to(device)
            assert indist_imgs.shape[0] == outdist_imgs.shape[0]

            # Compute adversarial out-distribution data
            model.eval()
            # Gradient ascent on the D model
            outdist_imgs_adv = perturb(model, outdist_imgs,
                                       normalization=normalization,
                                       **outdist_attack_config)
            # Gradient descent on the D model
            indist_input = perturb(model, indist_imgs,
                                   normalization=normalization,
                                   **indist_attack_config, descent=True)
            targets = torch.cat(
                [torch.ones(indist_imgs.shape[0], dtype=torch.float32),
                 torch.zeros(outdist_imgs_adv.shape[0],dtype=torch.float32)]).to(device)

            # Train the D model to seperate in-dist data and adversarial
            # out-distribution data
            set_train(model)
            optimizer.zero_grad()
            if args.r1reg > 0:
                indist_input.requires_grad_()
                pos_logits = forward(model, indist_input, normalization)
                reg_loss = r1_reg(pos_logits, indist_input)
                neg_logits = forward(model, outdist_imgs_adv, normalization)
                logits = torch.cat([pos_logits, neg_logits])
                loss = criterion(input=logits,
                                 target=targets) + args.r1reg * reg_loss
            else:
                pos_logits = forward(model, indist_input, normalization)
                neg_logits = forward(model, outdist_imgs_adv, normalization)
                logits = torch.cat([pos_logits, neg_logits])
                loss = criterion(input=logits, target=targets)
            loss.backward()
            optimizer.step()

            # AUC on in-dist data vs. adversarial out-distribution data
            fpr_, tpr_, thresholds = roc_curve(targets.data.cpu(),
                                               logits.data.cpu())
            auc = auc_fn(fpr_, tpr_)
            rolling_adv_auc.append(auc)

            # AUC on clean in-dist data vs. clean out-distribution data
            model.eval()
            with torch.no_grad():
                pos_logits = forward(model, indist_imgs, normalization)
                neg_logits = forward(model, outdist_imgs, normalization)
                logits = torch.cat([pos_logits, neg_logits])
                fpr_, tpr_, thresholds = roc_curve(targets.data.cpu(),
                                                   logits.data.cpu())
                clean_auc = auc_fn(fpr_, tpr_)

            rt = (pos_logits > 0).float().mean()
            rt0 = (neg_logits > 0).float().mean()
            perturbation = outdist_imgs_adv - outdist_imgs
            l2_dist = torch.norm(
                perturbation.reshape(perturbation.shape[0], -1), dim=1).mean().item()
            print(
                f'step {step} '
                f'step-size {outdist_attack_config["step_size"]} '
                f'epoch {epoch} iter {i}/{len(indist_loader)} '
                f'({datetime.now()}) '
                f'loss {loss:.3f} adv auc {auc:.3f} clean auc {clean_auc:.3f} '
                f'(100-avg adv auc {np.mean(rolling_adv_auc):.3f}) '
                f'dist {l2_dist:.3f}/{args.eps}/{step * args.step_size} '
                f'rt {rt:.3f} rt0 {rt0:.3f} '
                f'pos/neg {indist_imgs.shape[0]}/{outdist_imgs_adv.shape[0]}')

            log_interval = min(20, len(indist_loader))
            if i % log_interval == log_interval - 1:
                tb_logger.add_scalar('training_loss', loss.item(), global_step)
                tb_logger.add_scalar('adv_auc', auc, global_step)
                tb_logger.add_scalar('clean_auc', clean_auc, global_step)
                tb_logger.add_scalar('dist', l2_dist, global_step)
                tb_logger.add_scalar('rt', rt, global_step)
                tb_logger.add_scalar('rt0', rt0, global_step)
                tb_logger.add_scalar('step', step, global_step)

            # Save a checkpoint every 200 iterations or at the end of epoch
            if i % 200 == 199 or i == len(indist_loader) - 1:
                save_model(model, os.path.join(taskdir, 'model.pth'))
                save_model(optimizer, os.path.join(taskdir, 'optimizer.pth'))

            # If Adv AUC reaches a high value, interrupt training for this step
            if step != args.max_steps and \
                    len(rolling_adv_auc) == rolling_adv_auc.maxlen and \
                    np.mean(rolling_adv_auc) > args.AUC_th:
                print('step interrupted')
                step_interrupt = True
                break  # breaks the iteration loop

        if step_interrupt:
            break  # break the epoch loop
