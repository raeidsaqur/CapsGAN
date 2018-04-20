#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import argparse
import os, sys, random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
from tqdm import tqdm

import utils
import models.dcgan as dcgan
import models.mlp as mlp

isDebug = True
USE_CUDA = torch.cuda.is_available()
NUM_WORKERS = 4 * 1 if USE_CUDA else 2      # num_workers = 4 * NGPUs else 2

#default parameter values
DATASET = 'cifar10'
NETG_CIFAR10 = './samples/cifar10/netG_epoch_24.pth'
NETD_CIFAR10 = './samples/cifar10/netD_epoch_24.pth'
NETG_MNIST = './samples/mnist/netG_epoch_24.pth'
NETD_MNIST = './samples/mnist/netD_epoch_24.pth'


NUM_EPOCHS = 25
BATCH_SIZE = 128
IMG_SIZE = 64
IMG_CHANNELS = 3
NGF = BATCH_SIZE * 2
NDF = BATCH_SIZE * 2
LR_D = 0.00005
LR_G = 0.00005
D_ITERS = 5             # Number of D iterations per G iteration


def getOptimizers(opt, netG, netD):
    '''
    :param opt: Options
    :return: optimizerG, optimizerD (default RMSProp or ADAM)
    '''
    if opt.adam:
        if isDebug: print("Using ADAM Optimizer")
        optimizerD = optim.Adam(netD.parameters(), lr=opt.lrD, betas=(opt.beta1, 0.999))
        optimizerG = optim.Adam(netG.parameters(), lr=opt.lrG, betas=(opt.beta1, 0.999))
    else:
        if isDebug: print("Using RMSProp Optimizer")
        optimizerD = optim.RMSprop(netD.parameters(), lr = opt.lrD)
        optimizerG = optim.RMSprop(netG.parameters(), lr = opt.lrG)

    return optimizerG, optimizerD

def getNetworks(opt):
    '''
    Returns G, D
    :param opt: hyper-param options
    :return: (netG, netD)
    '''
    ngpu = int(opt.ngpu)
    nz = int(opt.nz)
    ngf = int(opt.ngf)
    ndf = int(opt.ndf)
    nc = int(opt.nc)
    n_extra_layers = int(opt.n_extra_layers)

    netG = __getGenerator(opt, ngpu, nz, ngf, ndf, nc, n_extra_layers)
    netD = __getDiscriminator(opt, ngpu, nz, ngf, ndf, nc, n_extra_layers)

    return netG, netD


def __getGenerator(opt, ngpu, nz, ngf, ndf, nc, n_extra_layers):
    if opt.noBN:
        if isDebug: print("Using No Batch Norm (DCGAN_G_nobn) for Generator")
        netG = dcgan.DCGAN_G_nobn(opt.imageSize, nz, nc, ngf, ngpu, n_extra_layers)
    elif opt.mlp_G:
        if isDebug: print("Using MLP_G for Generator")
        netG = mlp.MLP_G(opt.imageSize, nz, nc, ngf, ngpu)
    else:
        if isDebug: print("Using DCGAN_G for Generator")
        netG = dcgan.DCGAN_G(opt.imageSize, nz, nc, ngf, ngpu, n_extra_layers, bias=False)

    netG.apply(weights_init)
    if opt.netG != '': # load checkpoint if needed
        netG.load_state_dict(torch.load(opt.netG))
    print("netG:\n {0}".format(netG))

    return netG

def __getDiscriminator(opt, ngpu, nz, ngf, ndf, nc, n_extra_layers):
    if opt.mlp_D:
        if isDebug: print("Using MLP_D for Discriminator/Critic")
        netD = mlp.MLP_D(opt.imageSize, nz, nc, ndf, ngpu)
    else:
        if isDebug: print("Using DCGAN_D for Discriminator/Critic")
        netD = dcgan.DCGAN_D(opt.imageSize, nz, nc, ndf, ngpu, n_extra_layers, False)
        netD.apply(weights_init)

    if opt.netD != '':
        netD.load_state_dict(torch.load(opt.netD))
    print("netD:\n {0}".format(netD))

    return netD

def __getDataSet(opt):
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
        opt.imageSize = 32
        dataset = dset.MNIST(root=opt.dataroot, download=True, transform=transforms.Compose([
                                   transforms.Scale(opt.imageSize),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))
        # Update opt params for mnist
        if opt.load_dict:
            opt.netD = NETD_MNIST
            opt.netG = NETG_MNIST

    return dataset


def weights_init(m):
    '''
    Custom weights initialization called on netG and netD
    '''
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def main(opt):

    cudnn.benchmark = True
    opt.manualSeed = random.randint(1, 10000)  # fix seed
    print("Random Seed: ", opt.manualSeed)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)

    ## Path to generative samples storage
    if opt.experiment is None:
        opt.experiment = 'samples'
    os.system('mkdir {0}'.format(opt.experiment))

    if USE_CUDA and not opt.cuda:
        utils.eprint("WARNING: CUDA device available, please run with CUDA")

    dataset = __getDataSet(opt)
    assert dataset

    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=opt.batchSize,
                                             shuffle=True,
                                             num_workers=int(opt.workers))
    nz = int(opt.nz)
    nc = int(opt.nc)


    input = torch.FloatTensor(opt.batchSize, nc, opt.imageSize, opt.imageSize)
    noise = torch.FloatTensor(opt.batchSize, nz, 1, 1)
    fixed_noise = torch.FloatTensor(opt.batchSize, nz, 1, 1).normal_(0, 1)
    one = torch.FloatTensor([1])
    mone = one * -1

    ## Get Networks
    netG, netD = getNetworks(opt)
    if opt.cuda:
        if isDebug: print("Using CUDA")
        netD.cuda()
        netG.cuda()
        input = input.cuda()
        one, mone = one.cuda(), mone.cuda()
        noise, fixed_noise = noise.cuda(), fixed_noise.cuda()

    ## Setup Optimizers
    optimizerG, optimizerD = getOptimizers(opt, netG, netD)

    gen_iterations = 0
    for epoch in tqdm(range(opt.niter)):
        data_iter = iter(dataloader)
        i = 0
        while i < len(dataloader):
            ############################
            # (1) Update D network
            ###########################
            for p in netD.parameters():  # reset requires_grad
                p.requires_grad = True  # they are set to False below in netG update

            # train the discriminator Diters times
            if gen_iterations < 25 or gen_iterations % 500 == 0:
                Diters = 100
            else:
                Diters = opt.Diters
            j = 0
            while j < Diters and i < len(dataloader):
                j += 1

                # clamp parameters to a cube
                for p in netD.parameters():
                    p.data.clamp_(opt.clamp_lower, opt.clamp_upper)

                data = data_iter.next()
                i += 1

                # train with real
                real_cpu, _ = data
                netD.zero_grad()
                batch_size = real_cpu.size(0)

                if opt.cuda:
                    real_cpu = real_cpu.cuda()
                input.resize_as_(real_cpu).copy_(real_cpu)
                inputv = Variable(input)

                errD_real = netD(inputv)
                errD_real.backward(one)

                # train with fake
                noise.resize_(opt.batchSize, nz, 1, 1).normal_(0, 1)
                noisev = Variable(noise, volatile=True)  # totally freeze netG
                fake = Variable(netG(noisev).data)
                inputv = fake
                errD_fake = netD(inputv)
                errD_fake.backward(mone)
                errD = errD_real - errD_fake
                optimizerD.step()

            ############################
            # (2) Update G network
            ###########################
            for p in netD.parameters():
                p.requires_grad = False  # to avoid computation
            netG.zero_grad()
            # in case our last batch was the tail batch of the dataloader,
            # make sure we feed a full batch of noise
            noise.resize_(opt.batchSize, nz, 1, 1).normal_(0, 1)
            noisev = Variable(noise)
            fake = netG(noisev)
            errG = netD(fake)
            errG.backward(one)
            optimizerG.step()
            gen_iterations += 1

            print('[%d/%d][%d/%d][%d] Loss_D: %f Loss_G: %f Loss_D_real: %f Loss_D_fake %f'
                  % (epoch, opt.niter, i, len(dataloader), gen_iterations,
                     errD.data[0], errG.data[0], errD_real.data[0], errD_fake.data[0]))

            if gen_iterations % 500 == 0 or ((gen_iterations % 100 == 0) and (opt.dataset == 'mnist')):
                real_cpu = real_cpu.mul(0.5).add(0.5)
                vutils.save_image(real_cpu, '{0}/{1}/real_samples.png'.format(opt.experiment, opt.dataset))
                fake = netG(Variable(fixed_noise, volatile=True))
                fake.data = fake.data.mul(0.5).add(0.5)
                vutils.save_image(fake.data, '{0}/{1}/fake_samples_{2}.png'.format(opt.experiment, opt.dataset, gen_iterations))

        # do checkpointing
        if opt.niter > 25:
            if epoch % 10 == 0:
                torch.save(netG.state_dict(), '{0}/{1}/netG_epoch_{2}.pth'.format(opt.experiment, opt.dataset, epoch))
                torch.save(netD.state_dict(), '{0}/{1}/netD_epoch_{2}.pth'.format(opt.experiment, opt.dataset, epoch))
        else:
            torch.save(netG.state_dict(), '{0}/{1}/netG_epoch_{2}.pth'.format(opt.experiment, opt.dataset, epoch))
            torch.save(netD.state_dict(), '{0}/{1}/netD_epoch_{2}.pth'.format(opt.experiment, opt.dataset, epoch))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Pass configurations here")
    parser.add_argument('--dataroot', required=True, help='path to dataset')
    parser.add_argument('--dataset', required=False, type=str, default=DATASET, help='cifar10 | imagenet | folder | lfw ')
    parser.add_argument('--debug', default=False, help='True | False')
    parser.add_argument('--workers', type=int, default=NUM_WORKERS, help='number of data loading workers')
    parser.add_argument('--batchSize', type=int, default=BATCH_SIZE, help='input batch size')
    parser.add_argument('--imageSize', type=int, default=IMG_SIZE, help='the height / width of the input image to network')
    parser.add_argument('--nc', type=int, default=IMG_CHANNELS, help='input image channels')
    parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
    parser.add_argument('--ngf', type=int, default=NGF, help='number of generator features')
    parser.add_argument('--ndf', type=int, default=NDF, help='number of discriminator features')
    parser.add_argument('--niter', type=int, default=NUM_EPOCHS, help='number of epochs to train for')
    parser.add_argument('--lrD', type=float, default=LR_D, help='learning rate for Critic, default=0.00005')
    parser.add_argument('--lrG', type=float, default=LR_G, help='learning rate for Generator, default=0.00005')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
    parser.add_argument('--visualize', action='store_true', help='Enables Visdom')
    parser.add_argument('--cuda', action='store', default=None, type=int, help='Enables cuda')
    parser.add_argument('--load_dict', action='store_true', help='Loads saved state dicts')
    parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
    parser.add_argument('--netG', default='', help="path to netG (to continue training)")
    parser.add_argument('--netD', default='', help="path to netD (to continue training)")
    parser.add_argument('--clamp_lower', type=float, default=-0.01)
    parser.add_argument('--clamp_upper', type=float, default=0.01)
    parser.add_argument('--Diters', type=int, default=D_ITERS, help='number of D iters per each G iter')
    parser.add_argument('--noBN', action='store_true', help='use batchnorm or not (only for DCGAN)')
    parser.add_argument('--mlp_G', action='store_true', help='use MLP for G')
    parser.add_argument('--mlp_D', action='store_true', help='use MLP for D')
    parser.add_argument('--n_extra_layers', type=int, default=0, help='Number of extra layers on gen and disc')
    parser.add_argument('--experiment', default=None, help='Where to store samples and models')
    parser.add_argument('--adam', action='store_true', help='Whether to use adam (default is rmsprop)')
    opt = parser.parse_args()

    if opt.cuda is not None and opt.cuda >= 0:
        if torch.cuda.is_available():
            torch.cuda.set_device(opt.cuda)
            opt.cuda = True
        else:
            opt.cuda = False

    try:
        from eval.helper import *
        from eval.BLEU_score import *
        from visdom import Visdom
        import torchnet as tnt
        from torchnet.engine import Engine
        from torchnet.logger import VisdomPlotLogger, VisdomTextLogger, VisdomLogger
        canVisualize = True
    except ImportError as ie:
        print("Could not import vizualization imports. ", file=sys.stderr)
        canVisualize = False
    opt.visualize = True if (opt.visualize and canVisualize) else False

    main(opt)