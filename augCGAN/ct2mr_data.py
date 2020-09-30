import io
import os.path
import random

import numpy as np
import torch
import torch.utils.data
import torchvision.transforms as transforms
from PIL import Image

DEV_SIZE = 100

def load_ct2mr(root):
    """loads in memory numpy data files"""
    def _load(fname, type):
        arr = np.load(os.path.join(root, fname))
        if type=="MRI":
            # already normalized/standardized
            pass
        else:
            arr = arr / 1250. - 1.
        return arr.astype('float32')

    print "loading data numpy files..."
    trainA = _load("trainA.npy",'CT')
    trainB = _load("trainB.npy",'MRI')
    testA  = _load("valA.npy",'CT')
    testB  = _load("valB.npy",'MRI')
    print "done."

    # shuffle train data
    rand_state = random.getstate()
    random.seed(123)
    indx = range(len(trainA))
    random.shuffle(indx)
    trainA = trainA[indx]
    trainB = trainB[indx]
    random.setstate(rand_state)

    devA = trainA[:DEV_SIZE]
    devB = trainB[:DEV_SIZE]

    trainA = trainA[DEV_SIZE:]
    trainB = trainB[DEV_SIZE:]

    return trainA, trainB, devA, devB, testA, testB

def flip_horizontal(x):
    flips = [(slice(None, None, None),
              slice(None, None, None),
              slice(None, None, random.choice([-1, None])))
             for _ in xrange(x.shape[0])]
    return np.array([img[flip] for img, flip in zip(x, flips)])

class AlignedIterator(object):
    """Iterate multiple ndarrays (e.g. images and labels) IN THE SAME ORDER
    and return tuples of minibatches"""

    def __init__(self, data_A, data_B, **kwargs):
        super(AlignedIterator, self).__init__()

        assert data_A.shape[0] == data_B.shape[0], 'passed data differ in number!'
        self.data_A = data_A
        self.data_B = data_B
        self.num_samples = data_A.shape[0]

        batch_size = kwargs.get('batch_size', 100)
        shuffle = kwargs.get('shuffle', False)
        self.n_batches = self.num_samples / batch_size
        if self.num_samples % batch_size != 0:
            self.n_batches += 1

        self.batch_size = batch_size

        self.shuffle = shuffle
        self.reset()

    def __iter__(self):
        return self

    def reset(self):
        if self.shuffle:
            self.data_indices = np.random.permutation(self.num_samples)
        else:
            self.data_indices = np.arange(self.num_samples)
        self.batch_idx = 0

    def next(self):
        if self.batch_idx == self.n_batches:
            self.reset()
            raise StopIteration

        idx = self.batch_idx * self.batch_size
        chosen_indices = self.data_indices[idx:idx+self.batch_size]
        self.batch_idx += 1

        return {'A': torch.from_numpy(self.data_A[chosen_indices]),
                'B': torch.from_numpy(self.data_B[chosen_indices])}

    def __len__(self):
        return self.num_samples

class UnalignedIterator(object):
    """Iterate multiple ndarrays (e.g. several images) IN DIFFERENT ORDER
    and return tuples of minibatches"""
    def __init__(self, data_A, data_B, **kwargs):
        super(UnalignedIterator, self).__init__()

        assert data_A.shape[0] == data_B.shape[0], 'passed data differ in number!'
        self.data_A = data_A
        self.data_B = data_B

        self.num_samples = data_A.shape[0]

        self.batch_size = kwargs.get('batch_size', 100)
        self.n_batches = self.num_samples / self.batch_size
        if self.num_samples % self.batch_size != 0:
            self.n_batches += 1

        self.flip = True
        self.reset()

    def __iter__(self):
        return self

    def reset(self):
        self.data_indices = [np.random.permutation(self.num_samples) for _ in range(2)]
        self.batch_idx = 0

    def next(self):
        if self.batch_idx == self.n_batches:
            self.reset()
            raise StopIteration

        idx = self.batch_idx * self.batch_size
        chosen_indices_A = self.data_indices[0][idx:idx+self.batch_size]
        chosen_indices_B = self.data_indices[1][idx:idx+self.batch_size]

        self.batch_idx += 1

        if self.flip:
            self.data_A = flip_horizontal(self.data_A)
            self.data_B = flip_horizontal(self.data_B)

        return {'A': torch.from_numpy(self.data_A[chosen_indices_A]),
                'B': torch.from_numpy(self.data_B[chosen_indices_B])}

    def __len__(self):
        return self.num_samples


class Edges2Shoes(object):
    def __init__(self, opt, subset, unaligned, fraction, load_in_mem):
        self.root = opt.dataroot
        self.subset = subset
        self.unaligned = unaligned
        self.fraction = fraction
        self.load_in_mem = load_in_mem
        self.cropSize = opt.cropSize
        self.loadSize = (opt.loadSize, opt.loadSize)
        self.opt = opt
        assert fraction > 0. and fraction <= 1.
        if subset in ['dev', 'train']:
            self.dir_A = os.path.join(self.root, 'trainA')
            self.dir_B = os.path.join(self.root, 'trainB')
        elif subset == 'val':  # test set
            self.dir_A = os.path.join(self.root, 'valA')
            self.dir_B = os.path.join(self.root, 'valB')
        else:
            raise NotImplementedError('subset %s no supported' % subset)

        self.A_paths = sorted(make_dataset(self.dir_A))
        self.B_paths = sorted(make_dataset(self.dir_B))

        # shuffle data
        rand_state = random.getstate()
        random.seed(123)
        indx = range(len(self.A_paths))
        random.shuffle(indx)
        self.A_paths = [self.A_paths[i] for i in indx]
        self.B_paths = [self.B_paths[i] for i in indx]
        random.setstate(rand_state)
        if subset == "dev":
            self.A_paths = self.A_paths[:DEV_SIZE]
            self.B_paths = self.B_paths[:DEV_SIZE]
        elif subset == 'train':
            self.A_paths = self.A_paths[DEV_SIZE:]
            self.B_paths = self.B_paths[DEV_SIZE:]

        # return only fraction of the subset
        subset_size = int(len(self.A_paths) * fraction)
        self.A_paths = self.A_paths[:subset_size]
        self.B_paths = self.B_paths[:subset_size]

        if load_in_mem:
            mem_A_paths = []
            mem_B_paths = []
            for A, B in zip(self.A_paths, self.B_paths):
                with open(A, 'rb') as fa:
                    mem_A_paths.append(io.BytesIO(fa.read()))
                with open(B, 'rb') as fb:
                    mem_B_paths.append(io.BytesIO(fb.read()))
            self.A_paths = mem_A_paths
            self.B_paths = mem_B_paths

        self.A_size = len(self.A_paths)
        self.B_size = len(self.B_paths)

    def __getitem__(self, index):
        A_path = self.A_paths[index % self.A_size]
        if self.unaligned:
            index_B = random.randint(0, self.B_size - 1)
        else:
            index_B = index % self.A_size
        B_path = self.B_paths[index_B]
        A_img = Image.open(A_path).convert('I')
        B_img = Image.open(B_path).convert('I')

        params = get_params(self.cropSize, self.loadSize)
        transform = get_transform(self.opt, params)

        A_img = transform(A_img).type(torch.float32)
        B_img = transform(B_img).type(torch.float32)
        A_img = A_img / 1250. - 1.
        B_img = (B_img/1000.) - 1.
        return {'A': A_img, 'B': B_img}

    def __len__(self):
        return max(self.A_size, self.B_size)


class DataLoader(object):
    def __init__(self, opt, subset, unaligned, batchSize,
                 shuffle=False, fraction=1., load_in_mem=True, drop_last=False):
        self.opt = opt
        self.dataset = Edges2Shoes(opt, subset, unaligned, fraction, load_in_mem)
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=batchSize,
            shuffle=shuffle,
            num_workers=int(opt.nThreads),
            drop_last=drop_last)

    def load_data(self):
        return self.dataloader

    def __len__(self):
        return len(self.dataset)


def get_params(cropSize, loadSize):
    new_h, new_w = loadSize

    x = random.randint(0, np.maximum(0, new_w - cropSize))
    y = random.randint(0, np.maximum(0, new_h - cropSize))

    flip = random.random() > 0.5
    return {'crop_pos': (x, y), 'flip': flip}

def get_transform(opt, params):
    transform_list = [transforms.Resize([opt.loadSize, opt.loadSize], Image.NEAREST)]

    if opt.crop:
        transform_list.append(transforms.Lambda(lambda img: __crop(img, params['crop_pos'], opt.cropSize)))
        transform_list.append(transforms.Lambda(lambda img: __flip(img, params['flip'])))

    transform_list += [transforms.ToTensor()]
    return transforms.Compose(transform_list)

def __crop(img, pos, size):
    ow, oh = img.size
    x1, y1 = pos
    tw = th = size
    if (ow > tw or oh > th):
        return img.crop((x1, y1, x1 + tw, y1 + th))
    return img

def __flip(img, flip):
    if flip:
        return img.transpose(Image.FLIP_LEFT_RIGHT)
    return img

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