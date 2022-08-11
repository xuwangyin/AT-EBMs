from PIL import Image
from pathlib import Path
import torchvision.datasets as datasets
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--category', type=str, required=True)
parser.add_argument('--split', type=str, required=True)
parser.add_argument('--savedir', type=str, default=None)
parser.add_argument('--root', type=str, required=True)
args = parser.parse_args()

m = datasets.LSUN(root=args.root, classes=[f'{args.category}_{args.split}'])
if args.savedir is None:
    args.savedir = args.category
Path(f'{args.savedir}/{args.split}/data').mkdir(parents=True, exist_ok=True)
for i, data in tqdm(enumerate(m)):
    im = data[0]
    width, height = im.size   # Get dimensions
    if 256 not in [width, height]:
        ratio = max(256/width, 256/height)
        im = im.resize((int(width*ratio), int(height*ratio)))
    width, height = im.size   # Get dimensions
    left = (width - 256)/2
    right = (width + 256)/2
    top = (height - 256)/2
    bottom = (height + 256)/2
    # Crop the center of the image
    im = im.crop((left, top, right, bottom))
    im.save(f'{args.savedir}/{args.split}/data/{i+1:05d}.png')
