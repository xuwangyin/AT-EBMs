export CUDA_VISIBLE_DEVICES=0,1,2,3

args='--indist_aug --indist_steps 5
--r1reg 100 --optimizer adam --lr 0.00005
--batch_size 40 --step_size 2.0 --epochs 50 --pretrain
--dataset afhq256 --datadir ./datasets/'

python -u train.py $args --max_steps 25 --startstep 0 --logfid
