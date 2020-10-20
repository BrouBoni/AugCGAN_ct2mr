import argparse
import os.path

import numpy as np
from PIL import Image
from natsort import natsorted

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)

    return images

parser = argparse.ArgumentParser('create numpy data from image folders')
parser.add_argument('--root', help='data directory', type=str, default='/home/bbk/Documents/augmented_cyclegan-mr2ct/datasets/ct2mr_256')
parser.add_argument('--size', help='image size', type=int, default=256)
args = parser.parse_args()

root = args.root
size = args.size

for subset in ['val', 'train']:
    dir_A = os.path.join(root, '%sA' % subset)
    dir_B = os.path.join(root, '%sB' % subset)
    A_paths = sorted(make_dataset(dir_A))
    B_paths = sorted(make_dataset(dir_B))
    if subset == 'val':
        A_paths = natsorted(A_paths)
        B_paths = natsorted(B_paths)

    mem_A_np = []
    mem_B_np = []
    for i, (A, B) in enumerate(zip(A_paths, B_paths)):
        mem_A_np.append(np.expand_dims(np.asarray(Image.open(A).convert('I').resize((size,size), Image.BICUBIC)).clip(0,2500), axis=0))
        mem_B_np.append(np.expand_dims(np.asarray(Image.open(B).convert('I').resize((size,size), Image.BICUBIC)).clip(0,4000), axis=0))

    full_A = np.stack(mem_A_np)
    full_B = np.stack(mem_B_np)

    A_size = len(mem_A_np)
    B_size = len(mem_B_np)
    print("%sA size=%d" % (subset, A_size))
    print("%sB size=%d" % (subset, B_size))
    np.save(os.path.join(root, "%sA" % subset), full_A)
    np.save(os.path.join(root, "%sB" % subset), full_B)