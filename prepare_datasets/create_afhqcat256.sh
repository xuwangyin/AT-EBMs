wget -N https://www.dropbox.com/s/scckftx13grwmiv/afhq_v2.zip?dl=0 -O prepare_datasets/downloads/afhq_v2.zip
7z x prepare_datasets/downloads/afhq_v2.zip -oprepare_datasets/downloads/afhq_v2
python prepare_datasets/resize_crop.py --resize 256 --datadir prepare_datasets/downloads/afhq_v2/train/cat --savedir datasets/afhq256/train/cat/data
rm -rf prepare_datasets/downloads/afhq_v2
rm prepare_datasets/downloads/afhq_v2.zip

