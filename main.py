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

from utils import *
from dataloader import *
import models.dcgan as dcgan
import models.mlp as mlp
import models.capsgan as capsgan

isDebug = True
USE_CUDA = torch.cuda.is_available()
NUM_WORKERS = 4 * 1 if USE_CUDA else 2      # num_workers = 4 * NGPUs else 2

#default parameter values
DATASET = 'cifar10'
# NETG_CIFAR10 = './samples/cifar10/netG_epoch_24.pth'
# NETD_CIFAR10 = './samples/cifar10/netD_epoch_24.pth'
# NETG_MNIST = './samples/mnist/netG_epoch_24.pth'
# NETD_MNIST = './samples/mnist/netD_epoch_24.pth'



NUM_EPOCHS = 25 if not isDebug else 10
D_ITERS = 5 if not isDebug else 5            # Number of D iterations per G iteration
MAX_ITERS_PER_EPOCH = 100 if not isDebug else 100

BATCH_SIZE = 128
IMG_SIZE = 64
IMG_CHANNELS = 3
NGF = BATCH_SIZE * 2
NDF = BATCH_SIZE * 2
LR_D = 0.00005
LR_G = 0.00005


def getOptimizers(opt, netG, netD):
    '''
    :param opt: Options
    :return: optimizerG, optimizerD (default RMSProp or ADAM)
    '''
    if opt.adam or opt.caps_D:
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
    elif opt.caps_D:
        netG = dcgan.DCGAN_G(32, nz, nc, ngf, ngpu, n_extra_layers, bias=False, opt=opt)
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
    elif opt.caps_D:
        if isDebug: print("Using CapsNet for Discriminator/Critic")
        netD =capsgan.CapsNet(opt, num_classes=1)
    else:
        if isDebug: print("Using DCGAN_D for Discriminator/Critic")
        netD = dcgan.DCGAN_D(opt.imageSize, nz, nc, ndf, ngpu, n_extra_layers, False)
        netD.apply(weights_init)

    if opt.netD != '':
        netD.load_state_dict(torch.load(opt.netD))
    print("netD:\n {0}".format(netD))

    return netD


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
    cuda = opt.cuda; visualize = opt.visualize
    print(f"cuda = {cuda}, visualize = {opt.visualize}")
    if visualize:
        netD_loss_logger = VisdomPlotLogger('line', opts={'title': 'Discriminator (NetD) Loss'})
        netG_loss_logger = VisdomPlotLogger('line', opts={'title': 'Generator (NetG) Loss'})

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
        eprint("WARNING: CUDA device available, please run with CUDA")

    dataset = getDataSet(opt)
    assert dataset
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=opt.batchSize,
                                             shuffle=True,
                                             num_workers=int(opt.workers))

    if opt.dataset == 'mnist':
        train_dataset = dataset.train_dataset
        test_dataset = dataset.test_dataset
        dataset = dataset.full_dataset
        dataloader = torch.utils.data.DataLoader(train_dataset,
                                                 batch_size=opt.batchSize,
                                                 shuffle=True,
                                                 num_workers=int(opt.workers))
    print(f"len(dataloader) = {len(dataloader)}")
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
    all_data_size = len(dataloader)
    print(f"all_data_size i.e. len(dataloader) = {len(dataloader)}")
    for epoch in tqdm(range(opt.niter)):
        data_iter = iter(dataloader)
        i = 0
        # while i < len(dataloader):
        while i < MAX_ITERS_PER_EPOCH:
            ############################
            # (1) Train D network (netD)
            ###########################
            for p in netD.parameters():  # reset requires_grad
                p.requires_grad = True  # they are set to False below in netG update

            # train the discriminator Diters times
            # if gen_iterations < 25 or gen_iterations % 500 == 0:
            #     Diters = 100
            # else:
            #     Diters = opt.Diters
            Diters = opt.Diters
            print(f"Starting Training Discriminator of {Diters} iterations ... ")
            j = 0
            # while j < Diters and i < len(dataloader):
            while j < Diters and i < MAX_ITERS_PER_EPOCH:
                j += 1
                # clamp parameters to a cube
                for p in netD.parameters():
                    p.data.clamp_(opt.clamp_lower, opt.clamp_upper)

                data = data_iter.next()
                i += 1

                print("-" * 5 + "Train with real " + "-" * 5)
                netD.zero_grad()
                real_cpu, _ = data
                batch_size = real_cpu.size(0)
                y_real = Variable(torch.ones(batch_size))
                y_fake = Variable(torch.zeros(batch_size))
                if opt.cuda:
                    real_cpu = real_cpu.cuda()
                    # real_data = real_data.cuda()
                    y_real = y_real.cuda()
                    y_fake = y_fake.cuda()

                input.resize_as_(real_cpu).copy_(real_cpu)
                real_data = Variable(input)  # x_

                optimizerD.zero_grad()                          # clear accum. grads from prev batch
                V, reconstructions, masked = netD(real_data)
                netD_real_loss = netD.loss(data=real_data, x=V, target=y_real, reconstructions=reconstructions)
                print(f"\n\tepoch_{epoch}_batch_{j}_D_real_loss = {netD_real_loss}\n")
                z = torch.randn((batch_size, nz)).view(-1, nz, 1, 1)    # e.g. 128, 100, 1, 1
                z = Variable(z.cuda()) if opt.cuda else Variable(z)

                print("-" * 5 + "Train with fake " + "-" * 5)
                Gz = netG(z)
                Gz = Variable(Gz.data, volatile=True)
                #Gz = Variable(netG(z).data)
                # Fixed: here Gz (output of G is 128, 1, 32, 32) -> need (128, 1, 28, 28) for feeding to caps_D

                Vz, reconstructions_z, masked_z = netD(Gz)
                netD_fake_loss = netD.loss(data=Gz, x=Vz, target=y_fake, reconstructions=reconstructions_z)
                print(f"\tepoch_{epoch}_batch_{j}_D_fake_loss = {netD_fake_loss}")

                netD_train_loss = netD_real_loss + netD_fake_loss
                print(f"\tepoch_{epoch}_batch_{j}_D_train_loss = {netD_train_loss}")
                netD_train_loss.backward()
                optimizerD.step()

            ############################
            # (2) Update G network
            ###########################
            print(f"Starting Training Generator for {MAX_ITERS_PER_EPOCH-j} iterations ... ")

            netG.zero_grad(), optimizerG.zero_grad()
            netD.eval()
            # in case our last batch was the tail batch of the dataloader,
            # make sure we feed a full batch of noise
            z = torch.randn((opt.batchSize, nz)).view(-1, nz, 1, 1)
            z = Variable(z.cuda()) if opt.cuda else Variable(z)
            Gz = netG(z)
            Gz = Variable(Gz.data, volatile=True)
            Vz, reconstructions_z, masked_z = netD(Gz)

            netG_train_loss = netD.loss(data=Gz, x=Vz, target=y_real, reconstructions=reconstructions_z)
            netG_train_loss.backward()
            optimizerG.step()

            gen_iterations += 1
            print("\tgen_iterations: %d" % gen_iterations)
            print(f"\tepoch_{epoch}_batch_{gen_iterations}_G_train_loss = {netG_train_loss}")
            if visualize:
                netD_loss_logger.log(epoch, netD_train_loss)
                netG_loss_logger.log(epoch, netG_train_loss)

            if gen_iterations % 10 == 0 or ((gen_iterations % 100 == 0) and (opt.dataset == 'mnist')):
                real_cpu = real_cpu.mul(0.5).add(0.5)       #de-normalizing mnist to get real sample x_real = mu + sigma.z
                vutils.save_image(real_cpu, '{0}/{1}/real_samples.png'.format(opt.experiment, opt.dataset))
                fake = netG(Variable(fixed_noise, volatile=True))   # Get a sample from random data from the generator
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
    parser.add_argument('--caps_G', action='store_true', help='use CapsNet for G')
    parser.add_argument('--caps_D', action='store_true', help='use CapsNet for D')
    parser.add_argument('--n_extra_layers', type=int, default=0, help='Number of extra layers on gen and disc')
    parser.add_argument('--experiment', default=None, help='Where to store samples and models')
    parser.add_argument('--adam', action='store_true', help='Whether to use adam (default is rmsprop)')
    opt = parser.parse_args()

    if opt.cuda is not None and opt.cuda >= 0:
        if torch.cuda.is_available():
            torch.cuda.set_device(opt.cuda)
            opt.cuda = True
            torch.cuda.empty_cache()
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