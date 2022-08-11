export CUDA_VISIBLE_DEVICES=0

args='--cifar10_ood_detection --epochs 400 --optimizer sgd --lr 0.1
---step_size 0.1 -indist_eps 0.25 --eps 0.5 --indist_steps 10 --max_steps 20 --startstep 20
--batch-size 128 --dataset cifar10  --datadir ./datasets'

python -u train.py $args
