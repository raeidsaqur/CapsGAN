#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import os, sys
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

isDebug = False

NETG_CIFAR10 = './samples/cifar10/netG_epoch_24.pth'
NETD_CIFAR10 = './samples/cifar10/netD_epoch_24.pth'
NETG_MNIST = './samples/mnist/netG_epoch_24.pth'
NETD_MNIST = './samples/mnist/netD_epoch_24.pth'


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

    return dataset
