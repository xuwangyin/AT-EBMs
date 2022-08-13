# Learning Energy-Based Models With Adversarial Training
Paper link https://arxiv.org/abs/2012.06568

## Dependencies
```
torch==1.7.1
torchvision==0.8.2
numpy==1.21.5
scipy==1.7.3
Pillow==9.1.0
scikit_learn==1.0.2
tensorflow==1.15 # for FID evaluation
```

## Preparing Data

First download [support.tar](https://github-share.s3.amazonaws.com/support.tar). Put the file in the project directory and execute the command `tar xf support.tar`.
The support files include model checkpoints, third-party libraries, and some auxiliary data which are required for model training and reproducing experimental results.

By default, datasets are organized in the `./datasets` directory:

```
$ ls ./datasets
CelebAHQ256  imagenet256  AFHQ-png ...
```

### Target distribution dataset ( $p_\textrm{data}$ )

- **CelebA-HQ 256** Download [data512x512.zip](https://drive.google.com/drive/folders/11Vz0fqHS2rXDb5pprgTjpD7S2BAJhi1P) and unzip it to `./prepare_datasets/downloads/celebahq_files/`. Run the following command to create the dataset: `$ sh ./prepare_datasets/create_celebahq256.sh`

- **AFHQ-CAT 256** Run the command `sh ./prepare_datasets/create_afhqcat256.sh`

- **LSUN Church 256** 
  1. Download https://github.com/fyu/lsun to `prepare_datasets/downloads`
  2. Inside `prepare_datasets/downloads/lsun` run `python download.py -c church_outdoor; unzip church_outdoor_train_lmdb.zip` 
  3. Inside `prepare_datasets` run `python extract_png.py --root downloads/lsun --category church_outdoor --split train --savedir ../datasets/Church256`

### Out-of-distribution dataset ( $p_0$ )

- **The 80 million tiny images dataset (required for the CIFAR-10 task)**
Download [tiny_images.bin](http://www.archive.org/download/80-million-tiny-images-2-of-2/tiny_images.bin) to `./datasets`.

- **ImageNet (required for the 256x256 tasks)**
  Download ImageNet and organize it as follows (images do not need to be resized to 256x256):
  ```
  $ ls ./datasets/imagenet256
  ILSVRC2012_devkit_t12.tar.gz train val
  $ ls ./datasets/imagenet256/train
  n01440764  n01739381  n01978287  n02092002 ...
  $ ls datasets/imagenet256/val
  n01440764  n01739381  n01978287  n02092002 .. 
  ```



## Training Models

CIFAR-10 (generation)
> python -u train.py --indist_steps 5 --r1reg 0.01 --optimizer adam --lr 0.0005 --batch_size 32 --step_size 0.1 --epochs 5 --dataset cifar10 --datadir ./datasets' --max_steps 25 --startstep 0 --logfid

CelebA-HQ 256
> python -u train.py --indist_aug --indist_steps 5 --r1reg 30 --optimizer adam --lr 0.00005 --batch_size 40 --step_size 2.0 --epochs 5 --pretrain --dataset celebahq256 --datadir ./datasets --max_steps 40 --startstep 0 --logfid

AFHQ-CAT 256
> python -u train.py --indist_aug --indist_steps 5 --r1reg 100 --optimizer adam --lr 0.00005 --batch_size 40 --step_size 2.0 --epochs 50 --pretrain --dataset afhq256 --datadir ./datasets/' --max_steps 25 --startstep 0 --logfid

LSUN-Church 256
> python -u train.py --indist_aug --indist_steps 5 --r1reg 100 --optimizer adam --lr 0.00005 --batch_size 40 --step_size 2.0 --epochs 50 --pretrain --dataset afhq256 --datadir ./datasets --max_steps 25 --startstep 0 --logfid


## Reproduce Experimental Results

### Generation and FID evaluation
CIFAR-10: 
> python generate.py --dataset cifar10 --savedir eval_fid/cifar10-eval; python fid.py eval_fid/cifar10-eval data/fid_stats_cifar10_train.npz

CelebA-HQ 256: 
> python generate.py --dataset celebahq256 --savedir eval_fid/celebahq256-eval; python fid.py eval_fid/celebahq256-eval datasets/CelebAHQ256/train/data

AFHQ-CAT 256: 
> python generate.py --dataset afhqcat256 --savedir eval_fid/afhqcat256-eval; python fid.py eval_fid/afhqcat256-eval datasets/AFHQ-png/afhq256/train/cat/data/

LSUN-Church 256: 
> python generate.py --dataset church256 --savedir eval_fid/church256-eval; python fid.py eval_fid/church256-eval datasets/Church256/train/data/

### Worst-case out-of-distribution detection
CIFAR-10: 
> python eval_ood.py --task cifar10

CelebA-HQ 256: 
> python eval_ood.py --task celebahq256

AFHQ-CAT 256:
> python eval_ood.py --task afhqcat256

LSUN-Church 256: 
> python eval_ood.py --task church256


### 2D experiment
Use `2Dtasks/2DTasks.ipynb` to reproduce the 2D experiment results
