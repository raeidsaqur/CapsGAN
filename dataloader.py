#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import os, sys
import struct
import numpy as np
import matplotlib.pyplot as plt
import scipy.misc
from tqdm import tqdm
from os import makedirs
from os.path import join
from os.path import exists
from itertools import groupby

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim import Adam
from tqdm import tqdm
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms

from torch.utils.data import Dataset

isDebug = False

NETG_CIFAR10 = './samples/cifar10/netG_epoch_24.pth'
NETD_CIFAR10 = './samples/cifar10/netD_epoch_24.pth'
NETG_MNIST = './samples/mnist/netG_epoch_24.pth'
NETD_MNIST = './samples/mnist/netD_epoch_24.pth'
# --netG ./samples/smallnorb/netG_epoch_24.pth --netD ./samples/smallnorb/netD_epoch_24.pth

class SmallNORBSample:

    def __init__(self):
        self.image_lt  = None
        self.image_rt  = None
        self.category  = None
        self.instance  = None
        self.elevation = None
        self.azimuth   = None
        self.lighting  = None

    def __lt__(self, other):
        return self.category < other.category or \
                (self.category == other.category and self.instance < other.instance)

    def show(self, subplots):
        fig, axes = subplots
        fig.suptitle(
            'Category: {:02d} - Instance: {:02d} - Elevation: {:02d} - Azimuth: {:02d} - Lighting: {:02d}'.format(
                self.category, self.instance, self.elevation, self.azimuth, self.lighting))
        axes[0].imshow(self.image_lt, cmap='gray')
        axes[1].imshow(self.image_rt, cmap='gray')

    @property
    def pose(self):
        # e.g. 9 * 18 * 6
        return np.array([self.elevation, self.azimuth, self.lighting], dtype=np.float32)


class SmallNORBDataset(Dataset):
    # Number of examples in both train and test set
    n_examples = 24300

    def __init__(self, opt):
        """
        Initialize small NORB dataset wrapper

        Parameters
        ----------
        dataset_root: str
            Path to directory where small NORB archives have been extracted.
        """

        # self.dataset_root = dataset_root
        self.opt = opt
        self.dataset_root = opt.dataroot
        self.initialized = False
        self.batch_size = opt.batchSize

        self.transform = transforms.Compose([
            transforms.Resize((opt.imageSize, opt.imageSize)),
            transforms.ToTensor(),
            # transforms.Normalize(mean=(0.5), std=(0.5)),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        # Categories present in small NORB dataset
        self.categories = ['animals', 'humans', 'airplanes', 'trucks', 'cars']
        cat_to_ix = {}
        for i, cat in enumerate(self.categories):
            cat_to_ix[cat] = i
        self.cat_to_ix = cat_to_ix
        self.ix_to_cat = {v:k for (k,v) in self.cat_to_ix.items()}

        self.cat_to_images = None

        if not os.path.isfile("data/smallnorb/cat_2_images.pth"):
            # Store path for each file in small NORB dataset (for compatibility the original filename is kept)
            self.dataset_files = {
                'train': {
                    'cat': join(self.dataset_root, 'smallnorb-5x46789x9x18x6x2x96x96-training-cat.mat'),
                    'info': join(self.dataset_root, 'smallnorb-5x46789x9x18x6x2x96x96-training-info.mat'),
                    'dat': join(self.dataset_root, 'smallnorb-5x46789x9x18x6x2x96x96-training-dat.mat')
                },
                'test': {
                    'cat': join(self.dataset_root, 'smallnorb-5x01235x9x18x6x2x96x96-testing-cat.mat'),
                    'info': join(self.dataset_root, 'smallnorb-5x01235x9x18x6x2x96x96-testing-info.mat'),
                    'dat': join(self.dataset_root, 'smallnorb-5x01235x9x18x6x2x96x96-testing-dat.mat')
                }
            }

            # Initialize both train and test data structures
            self.data = {
                'train': [SmallNORBSample() for _ in range(SmallNORBDataset.n_examples)],
                'test': [SmallNORBSample() for _ in range(SmallNORBDataset.n_examples)]
            }

            # Fill data structures parsing dataset binary files
            for data_split in ['train', 'test']:
                self._fill_data_structures(data_split)

            self.cat_to_images = self._get_cat_to_images()

        else:
            self.cat_to_images  = torch.load("data/smallnorb/cat_2_images.pth")

        self.idx = 0
        self.initialized = True

    def __len__(self):
        if self.cat_to_images:
            self.max_length = len(self.cat_to_images[self.ix_to_cat[0]])
        else:
            self.max_length = 24300
        return self.max_length

    def __iter__(self):
        return self

    # def __next__(self):
    #     return self.next()

    def reset(self):
        self.idx = 0
        # random.shuffle(self.lines)

    def __getitem__(self, idx):
        return self.next()

    def next(self, cat_idx=4):
        img_start_idx = self.idx
        # img_end_idx = self.idx + self.batch_size
        img_end_idx = self.idx + 1
        # iterator edge case
        if img_end_idx  >= self.max_length:
            raise StopIteration

        # increment idx (bookeeping for iterator)
        # self.idx += self.batch_size
        self.idx += 1
        cat_images = self.cat_to_images[self.ix_to_cat[cat_idx]][img_start_idx: img_end_idx]
        for cat_image in cat_images:

            cat_image = cat_image[16:80, 16:80]
            # print(f"cat_image.shape = {cat_image.shape}")
            channels = self.opt.nc

            # Note the discrepancy of C,H,W and H,W,C for transforming
            # ToPILImage from Tensor vs. ndarray.
            # cat_image = np.expand_dims(cat_image, axis=0)
            cat_image = cat_image[:, :, np.newaxis]
            H, W, C = cat_image.shape

            cat_image = transforms.ToPILImage()(cat_image)
            # input_data = torch.ByteTensor(channels, H, W).random_(0, 255).float().div_(255)
            if self.transform:
                cat_image = self.transform(cat_image)
                # cat_image.resize_( (1, 1, 1, 64, 64))
                # cat_image = torch.squeeze(cat_image, 0)
                # cat_image = torch.squeeze(cat_image, 0)

            # cat_image = transforms.Resize(cat_image, size=self.opt.imageSize)

        # input_target = cat_idx * np.ones((1, 1))
        input_target = int(cat_idx)

        # convert to torch long tensor
        # input_data = torch.from_numpy(np.asarray(cat_images)).long()
        # target_data = torch.from_numpy(np.asarray(input_target)).long()
        # input_data = transforms.Resize(input_data, size=self.opt.imageSize)

        return cat_image, input_target



    def _get_cat_to_images(self):


        cat_0 = []
        cat_1 = []
        cat_2 = []
        cat_3 = []
        cat_4 = []

        train_dataset = self.group_dataset_by_category_and_instance(dataset_split="train")
        for categories in train_dataset:
            for sample in categories:
                # SmallNorbSample
                cat = sample.category
                cat_list = cat_0
                if cat == 0:
                    cat_list = cat_0
                elif cat == 1:
                    cat_list = cat_1
                elif cat == 2:
                    cat_list = cat_2
                elif cat == 3:
                    cat_list = cat_3
                elif cat == 4:
                    cat_list = cat_4
                cat_list.append(sample.image_lt)
                cat_list.append(sample.image_rt)

        cat_2_images = {'animals': cat_0,
                        'humans': cat_1,
                        'airplanes': cat_2,
                        'trucks': cat_3,
                        'cars': cat_4}

        torch.save(cat_2_images, "data/smallnorb/cat_2_images.pth")

    def explore_random_examples(self, dataset_split):
        """
        Visualize random examples for dataset exploration purposes

        Parameters
        ----------
        dataset_split: str
            Dataset split, can be either 'train' or 'test'

        Returns
        -------
        None
        """
        if self.initialized:
            subplots = plt.subplots(nrows=1, ncols=2)
            for i in np.random.permutation(SmallNORBDataset.n_examples):
                self.data[dataset_split][i].show(subplots)
                plt.waitforbuttonpress()
                plt.cla()

    def export_to_img(self, export_dir, imformat="png"):
        """
        Export all dataset images to `export_dir` directory

        Parameters
        ----------
        export_dir: str
            Path to export directory (which is created if nonexistent)

        Returns
        -------
        None
        """
        if self.initialized:
            print('Exporting images to {}...'.format(export_dir), end='', flush=True)
            for split_name in ['train', 'test']:

                split_dir = join(export_dir, split_name)
                if not exists(split_dir):
                    makedirs(split_dir)

                for i, norb_example in enumerate(self.data[split_name]):
                    category = SmallNORBDataset.categories[norb_example.category]
                    instance = norb_example.instance

                    image_lt_path = join(split_dir, '{:06d}_{}_{:02d}_lt.{}'.format(i, category, instance, imformat))
                    image_rt_path = join(split_dir, '{:06d}_{}_{:02d}_rt.{}'.format(i, category, instance, imformat))

                    scipy.misc.imsave(image_lt_path, norb_example.image_lt)
                    scipy.misc.imsave(image_rt_path, norb_example.image_rt)
            print('Done.')

    def group_dataset_by_category_and_instance(self, dataset_split):
        """
        Group small NORB dataset for (category, instance) key

        Parameters
        ----------
        dataset_split: str
            Dataset split, can be either 'train' or 'test'

        Returns
        -------
        groups: list
            List of 25 groups of 972 elements each. All examples of each group are
            from the same category and instance
        """
        if dataset_split not in ['train', 'test']:
            raise ValueError('Dataset split "{}" not allowed.'.format(dataset_split))

        groups = []
        for key, group in groupby(iterable=sorted(self.data[dataset_split]),
                                  key=lambda x: (x.category, x.instance)):
            groups.append(list(group))

        return groups

    def _fill_data_structures(self, dataset_split):
        """
        Fill SmallNORBDataset data structures for a certain `dataset_split`.

        This means all images, category and additional information are loaded from binary
        files of the current split.

        Parameters
        ----------
        dataset_split: str
            Dataset split, can be either 'train' or 'test'

        Returns
        -------
        None

        """
        dat_data = self._parse_NORB_dat_file(self.dataset_files[dataset_split]['dat'])
        cat_data = self._parse_NORB_cat_file(self.dataset_files[dataset_split]['cat'])
        info_data = self._parse_NORB_info_file(self.dataset_files[dataset_split]['info'])
        for i, small_norb_example in enumerate(self.data[dataset_split]):
            small_norb_example.image_lt = dat_data[2 * i]
            small_norb_example.image_rt = dat_data[2 * i + 1]
            small_norb_example.category = cat_data[i]
            small_norb_example.instance = info_data[i][0]
            small_norb_example.elevation = info_data[i][1]
            small_norb_example.azimuth = info_data[i][2]
            small_norb_example.lighting = info_data[i][3]

    @staticmethod
    def matrix_type_from_magic(magic_number):
        """
        Get matrix data type from magic number
        See here: https://cs.nyu.edu/~ylclab/data/norb-v1.0-small/readme for details.

        Parameters
        ----------
        magic_number: tuple
            First 4 bytes read from small NORB files

        Returns
        -------
        element type of the matrix
        """
        convention = {'1E3D4C51': 'single precision matrix',
                      '1E3D4C52': 'packed matrix',
                      '1E3D4C53': 'double precision matrix',
                      '1E3D4C54': 'integer matrix',
                      '1E3D4C55': 'byte matrix',
                      '1E3D4C56': 'short matrix'}
        magic_str = bytearray(reversed(magic_number)).hex().upper()
        return convention[magic_str]

    @staticmethod
    def _parse_small_NORB_header(file_pointer):
        """
        Parse header of small NORB binary file

        Parameters
        ----------
        file_pointer: BufferedReader
            File pointer just opened in a small NORB binary file

        Returns
        -------
        file_header_data: dict
            Dictionary containing header information
        """
        # Read magic number
        magic = struct.unpack('<BBBB', file_pointer.read(4))  # '<' is little endian)

        # Read dimensions
        dimensions = []
        num_dims, = struct.unpack('<i', file_pointer.read(4))  # '<' is little endian)
        for _ in range(num_dims):
            dimensions.extend(struct.unpack('<i', file_pointer.read(4)))

        file_header_data = {'magic_number': magic,
                            'matrix_type': SmallNORBDataset.matrix_type_from_magic(magic),
                            'dimensions': dimensions}
        return file_header_data

    @staticmethod
    def _parse_NORB_cat_file(file_path):
        """
        Parse small NORB category file

        Parameters
        ----------
        file_path: str
            Path of the small NORB `*-cat.mat` file

        Returns
        -------
        examples: ndarray
            Ndarray of shape (24300,) containing the category of each example
        """
        with open(file_path, mode='rb') as f:
            header = SmallNORBDataset._parse_small_NORB_header(f)

            num_examples, = header['dimensions']

            struct.unpack('<BBBB', f.read(4))  # ignore this integer
            struct.unpack('<BBBB', f.read(4))  # ignore this integer

            examples = np.zeros(shape=num_examples, dtype=np.int32)
            for i in tqdm(range(num_examples), desc='Loading categories...'):
                category, = struct.unpack('<i', f.read(4))
                examples[i] = category

            return examples

    @staticmethod
    def _parse_NORB_dat_file(file_path):
        """
        Parse small NORB data file

        Parameters
        ----------
        file_path: str
            Path of the small NORB `*-dat.mat` file

        Returns
        -------
        examples: ndarray
            Ndarray of shape (48600, 96, 96) containing images couples. Each image couple
            is stored in position [i, :, :] and [i+1, :, :]
        """
        with open(file_path, mode='rb') as f:
            header = SmallNORBDataset._parse_small_NORB_header(f)

            num_examples, channels, height, width = header['dimensions']

            examples = np.zeros(shape=(num_examples * channels, height, width), dtype=np.uint8)

            for i in tqdm(range(num_examples * channels), desc='Loading images...'):
                # Read raw image data and restore shape as appropriate
                image = struct.unpack('<' + height * width * 'B', f.read(height * width))
                image = np.uint8(np.reshape(image, newshape=(height, width)))

                examples[i] = image

        return examples

    @staticmethod
    def _parse_NORB_info_file(file_path):
        """
        Parse small NORB information file

        Parameters
        ----------
        file_path: str
            Path of the small NORB `*-info.mat` file

        Returns
        -------
        examples: ndarray
            Ndarray of shape (24300,4) containing the additional info of each example.

             - column 1: the instance in the category (0 to 9)
             - column 2: the elevation (0 to 8, which mean cameras are 30, 35,40,45,50,55,60,65,70
               degrees from the horizontal respectively)
             - column 3: the azimuth (0,2,4,...,34, multiply by 10 to get the azimuth in degrees)
             - column 4: the lighting condition (0 to 5)
        """
        with open(file_path, mode='rb') as f:

            header = SmallNORBDataset._parse_small_NORB_header(f)

            struct.unpack('<BBBB', f.read(4))  # ignore this integer

            num_examples, num_info = header['dimensions']

            examples = np.zeros(shape=(num_examples, num_info), dtype=np.int32)

            for r in tqdm(range(num_examples), desc='Loading info...'):
                for c in range(num_info):
                    info, = struct.unpack('<i', f.read(4))
                    examples[r, c] = info

        return examples


class Mnist:
    def __init__(self, opt):
        dataset_transform = transforms.Compose([
            transforms.Resize(opt.imageSize),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        self.full_dataset = dset.MNIST(root=opt.dataroot, download=True, transform=dataset_transform)
        self.train_dataset = dset.MNIST(opt.dataroot, train=True, download=True, transform=dataset_transform)
        self.test_dataset = dset.MNIST(opt.dataroot, train=False, download=True, transform=dataset_transform)
        self.train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=opt.batchSize, shuffle=True)
        self.test_loader = torch.utils.data.DataLoader(self.test_dataset, batch_size=opt.batchSize, shuffle=True)














def getDataSet(opt):
    if isDebug: print(f"Getting dataset: {opt.dataset} ... ")

    dataset = None
    if opt.dataset in ['imagenet', 'folder', 'lfw']:
        # folder dataset
        traindir = os.path.join(opt.dataroot, f"{opt.dataroot}/train")
        valdir = os.path.join(opt.dataroot, f"{opt.dataroot}/val")
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        train_dataset = dset.ImageFolder(
            traindir,
            transforms.Compose([
                transforms.RandomResizedCrop(opt.imageSize),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]))

        dataset = dset.ImageFolder(root=opt.dataroot,
                                   transform=transforms.Compose([
                                       transforms.Scale(opt.imageSize),
                                       transforms.CenterCrop(opt.imageSize),
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                   ]))
    elif opt.dataset == 'lsun':
        dataset = dset.LSUN(db_path=opt.dataroot, classes=['bedroom_train'],
                            transform=transforms.Compose([
                                transforms.Scale(opt.imageSize),
                                transforms.CenterCrop(opt.imageSize),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                            ]))
    elif opt.dataset == 'cifar10':
        dataset = dset.CIFAR10(root=opt.dataroot, download=True,
                               transform=transforms.Compose([
                                   transforms.Scale(opt.imageSize),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))
        # Load pre-trained state dict
        if opt.load_dict:
            opt.netD = NETD_CIFAR10
            opt.netG = NETG_CIFAR10
    elif opt.dataset == 'mnist':
        opt.nc = 1
        # opt.imageSize = 32
        # dataset = dset.MNIST(root=opt.dataroot, download=True, transform=transforms.Compose([
        #                            transforms.Scale(opt.imageSize),
        #                            transforms.ToTensor(),
        #                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        #                        ]))
        dataset = Mnist(opt)
        # Update opt params for mnist
        if opt.load_dict:
            opt.netD = NETD_MNIST
            opt.netG = NETG_MNIST

    elif opt.dataset == 'smallnorb':
        opt.nc = 1
        # opt.imageSize = 32
        # dataset = dset.MNIST(root=opt.dataroot, download=True, transform=transforms.Compose([
        #                            transforms.Scale(opt.imageSize),
        #                            transforms.ToTensor(),
        #                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        #                        ]))
        dataset = SmallNORBDataset(opt)
        # Update opt params for smallnorb



    return dataset
