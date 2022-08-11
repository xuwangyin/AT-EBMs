# Learning Energy-Based Models With Adversarial Training

## Dependencies

See `requirements.txt`

## Preparing Data

First download [support.tar](https://github-share.s3.amazonaws.com/support.tar). Put the file in the project directory and execute the command `tar xf support.tar`.
The support files include model checkpoints, third-party libraries, and some auxiliary data which are required for model training and reproducing experimental results.

By default, datasets are organized in the `./datasets` directory:

```
$ ls ./datasets
CelebAHQ256  imagenet256  AFHQ-png ...
```

### Out-of-distribution datasets ($p_0$ dataset)

**The 80 million tiny images dataset**

Download [tiny_images.bin](http://www.archive.org/download/80-million-tiny-images-2-of-2/tiny_images.bin) to `./datasets`.

**ImageNet**

Download ImageNet and organize it as follows (images do not need to be resized to 256x256):

```
$ ls ./datasets/imagenet256
ILSVRC2012_devkit_t12.tar.gz train val
$ ls ./datasets/imagenet256/train
n01440764  n01739381  n01978287  n02092002 ...
$ ls datasets/imagenet256/val
n01440764  n01739381  n01978287  n02092002 .. 
```
### Target distribution datasets ($p_\textrm{data}$ dataset)

**CelebA-HQ 256**

Download [data512x512.zip](https://drive.google.com/drive/folders/11Vz0fqHS2rXDb5pprgTjpD7S2BAJhi1P) (the 512x512 version of CelebA-HQ dataset) and unzip it to `./prepare_datasets/downloads/celebahq_files/`. Run the following command to create the dataset: `$ sh ./prepare_datasets/create_celebahq256.sh`

**AFHQ-CAT 256**

Run the command `sh ./prepare_datasets/create_afhqcat256.sh`

**LSUN Church 256**
1. Download https://github.com/fyu/lsun to `prepare_datasets/downloads`
2. Inside `prepare_datasets/downloads/lsun` run `python download.py -c church_outdoor; unzip church_outdoor_train_lmdb.zip` 
3. Inside `prepare_datasets` run `python extract_png.py --root downloads/lsun --category church_outdoor --split train --savedir ../datasets/Church256`



## Training Models

CIFAR-10 (generation)
```bash
#!/bin/bash
args='--indist_steps 5 --r1reg 0.01 --optimizer adam --lr 0.0005
--batch_size 32 --step_size 0.1 --epochs 5
--dataset cifar10 --datadir ./datasets'

python -u train.py $args --max_steps 25 --startstep 0 --logfid
```

CelebA-HQ 256
```bash
#!/bin/bash
args='--indist_aug --indist_steps 5
--r1reg 30 --optimizer adam --lr 0.00005
--batch_size 40 --step_size 2.0 --epochs 5 --pretrain
--dataset celebahq256 --datadir ./datasets'

python -u train.py $args --max_steps 40 --startstep 0 --logfid
```

AFHQ-CAT 256
```bash
#!/bin/bash
args='--indist_aug --indist_steps 5
--r1reg 100 --optimizer adam --lr 0.00005
--batch_size 40 --step_size 2.0 --epochs 50 --pretrain
--dataset afhq256 --datadir ./datasets/'

python -u train.py $args --max_steps 25 --startstep 0 --logfid
```

LSUN-Church 256
```bash
#!/bin/bash
args='--indist_aug --indist_steps 5
--r1reg 100 --optimizer adam --lr 0.00005
--batch_size 40 --step_size 2.0 --epochs 50 --pretrain
--dataset afhq256 --datadir ./datasets/'

python -u train.py $args --max_steps 25 --startstep 0 --logfid
```


## Reproduce Experimental Results

### Generation and FID evaluation
CIFAR-10: `python generate.py --dataset cifar10 --savedir eval_fid/cifar10-gen; python fid.py eval_fid/cifar10-gen data/fid_stats_cifar10_train.npz`
CelebA-HQ 256: `python generate.py --dataset celebahq256 --savedir eval_fid/celebahq256-gen; python fid.py eval_fid/celebahq256-gen datasets/CelebAHQ256/train/data`
AFHQ-CAT 256: `python generate.py --dataset afhqcat256 --savedir eval_fid/afhqcat256-gen; python fid.py eval_fid/afhqcat256-gen datasets/AFHQ-png/afhq256/train/cat/data/`
LSUN-Church 256: `python generate.py --dataset church256 --savedir eval_fid/church256-gen; python fid.py eval_fid/church256-gen datasets/Church256/train/data/`

### Worst-case out-of-distribution detection

CIFAR-10: `python eval_ood_cifar10.py --dataset CIFAR100` (change the --dataset parameter to obtain results on other OOD datasets)
CelebA-HQ 256: `python eval_ood_dataset256.py --task celebahq256`
AFHQ-CAT 256: `python eval_ood_dataset256.py --task afhqcat256`
LSUN-Church 256: `python eval_ood_dataset256.py --task church256`


### 2D experiment
Use `2Dtasks/2DTasks.ipynb` to reproduce the 2D experiment results
