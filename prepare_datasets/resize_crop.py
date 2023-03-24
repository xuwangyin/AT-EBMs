import numpy as np
import os
from PIL import Image
from pathlib import Path
import argparse
import glob


parser = argparse.ArgumentParser()
parser.add_argument('--size', type=int, required=True)
parser.add_argument('--datadir', type=str, required=True)
parser.add_argument('--savedir', type=str, required=True)
parser.add_argument('--crop', action='store_true')
args = parser.parse_args()

i = 0
files = glob.glob(os.path.join(args.datadir, '**/*.*'), recursive=True)
print('found {} files'.format(len(files)))
for fullpath in files:
  fullpath = fullpath.strip()
  savefile = fullpath.replace(args.datadir, args.savedir)
  if not savefile.endswith('.png'):
      savefile += '.png'  
  Path(os.path.dirname(savefile)).mkdir(parents=True, exist_ok=True)
  
  im = Image.open(fullpath)
  # Resize the image to have a minimum dimension of the specified size
  width, height = im.size   # Get dimensions
  ratio = max(args.size/width, args.size/height)
  im = im.resize((int(width*ratio), int(height*ratio)), Image.LANCZOS)
  if args.crop:
    width, height = im.size   # Get dimensions
    left = (width - args.size)/2
    right = (width + args.size)/2
    top = (height - args.size)/2
    bottom = (height + args.size)/2
    # Crop the center of the image
    im = im.crop((left, top, right, bottom))
  im.save(savefile)

  if i % 100 == 0:
     print('{} -> {} ({}%)'.format(fullpath, savefile, i*100/len(files)))
  i += 1
