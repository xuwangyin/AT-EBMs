export CUDA_VISIBLE_DEVICES=0

args='--indist_steps 5 --r1reg 0.01 --optimizer adam --lr 0.0005
--batch_size 32 --step_size 0.1 --epochs 5
--dataset cifar10 --datadir ./datasets'

python -u train.py $args --max_steps 25 --startstep 0 --logfid
