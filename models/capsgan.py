#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim import Adam
from torchvision import datasets, transforms
from tqdm import tqdm
from dataloader import *
from utils import *


isDebug = True
USE_CUDA = torch.cuda.is_available()
BATCH_SIZE = 128

class ConvLayer(nn.Module):
    def __init__(self, in_channels=1, out_channels=256, kernel_size=9):
        super(ConvLayer, self).__init__()

        self.conv = nn.Conv2d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size,
                              stride=1
                              )

    def forward(self, x):
        return F.relu(self.conv(x))


class PrimaryCaps(nn.Module):
    def __init__(self, num_capsules=8, in_channels=256, out_channels=32, kernel_size=9, opt=None):
        super(PrimaryCaps, self).__init__()
        # Holds 8 PrimaryCaps Conv2d modules
        self.capsules = nn.ModuleList([
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=2, padding=0)
            for _ in range(num_capsules)])

    def forward(self, x):
        if isDebug: print(f"PrimaryCaps.forward(x): x.size = {x.size(0)} by {x.size(1)}")
        u = [capsule(x) for capsule in self.capsules]
        u = torch.stack(u, dim=1)
        u = u.view(x.size(0), 32 * 6 * 6, -1)
        return self.squash(u)

    def squash(self, input_tensor):
        squared_norm = (input_tensor ** 2).sum(-1, keepdim=True)
        output_tensor = squared_norm * input_tensor / ((1. + squared_norm) * torch.sqrt(squared_norm))
        return output_tensor


class DigitCaps(nn.Module):
    def __init__(self, num_capsules=10, num_routes=32 * 6 * 6, in_channels=8, out_channels=16, opt=None):
        super(DigitCaps, self).__init__()

        self.in_channels = in_channels
        self.num_routes = num_routes
        self.num_capsules = num_capsules

        self.W = nn.Parameter(torch.randn(1, num_routes, num_capsules, out_channels, in_channels))

    def forward(self, x):
        if isDebug: print(f"DigitCaps.forward(x): x.size = {x.size(0)} by {x.size(1)}")
        batch_size = x.size(0)
        x = torch.stack([x] * self.num_capsules, dim=2).unsqueeze(4)

        W = torch.cat([self.W] * batch_size, dim=0)
        u_hat = torch.matmul(W, x)

        b_ij = Variable(torch.zeros(1, self.num_routes, self.num_capsules, 1))
        if USE_CUDA:
            b_ij = b_ij.cuda()

        # Dynamic Routing
        num_iterations = 3  # Just a huper parameter
        for iteration in range(num_iterations):
            # UserWarning: Implicit dimension choice for softmax has been deprecated. Change the call to include dim=X as an argument.
            # c_ij = F.softmax(b_ij)
            c_ij = F.softmax(b_ij, dim=-1)
            c_ij = torch.cat([c_ij] * batch_size, dim=0).unsqueeze(4)

            s_j = (c_ij * u_hat).sum(dim=1, keepdim=True)
            v_j = self.squash(s_j)

            if iteration < num_iterations - 1:
                a_ij = torch.matmul(u_hat.transpose(3, 4), torch.cat([v_j] * self.num_routes, dim=1))
                b_ij = b_ij + a_ij.squeeze(4).mean(dim=0, keepdim=True)

        return v_j.squeeze(1)

    def squash(self, input_tensor):
        squared_norm = (input_tensor ** 2).sum(-1, keepdim=True)
        output_tensor = squared_norm * input_tensor / ((1. + squared_norm) * torch.sqrt(squared_norm))
        return output_tensor


class Decoder(nn.Module):
    def __init__(self, opt=None, num_classes=10):
        super(Decoder, self).__init__()
        self.num_classes = num_classes
        final_fc_layer = nn.Linear(1024, (opt.imageSize * opt.imageSize)) if opt is not None else nn.Linear(1024, 784)
        self.reconstruction_layers = nn.Sequential(
            nn.Linear((16 * num_classes), 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            # nn.Linear(1024, 784),
            final_fc_layer,
            nn.Sigmoid()
        )

    def forward(self, x, data):
        '''
        :param x: activation vector (length 16)
        :param data: data (for e.g. 28 by 28 mnist)
        :return: reconstructions, masked
        '''
        if isDebug: print(f"Decoder.forward(x): x.size = {x.size(0)} by {x.size(1)}")
        classes = torch.sqrt((x ** 2).sum(2))

        # UserWarning: Implicit dimension choice for softmax has been deprecated. Change the call to include dim=X as an argument.
        # classes = F.softmax(classes)
        classes = F.softmax(classes, dim=-1)
        _, max_length_indices = classes.max(dim=1)
        # masked = Variable(torch.sparse.torch.eye(10))
        masked = Variable(torch.sparse.torch.eye(self.num_classes))
        if USE_CUDA:
            masked = masked.cuda()
        masked = masked.index_select(dim=0, index=Variable(max_length_indices.squeeze(1).data))

        reconstructions = self.reconstruction_layers((x * masked[:, :, None, None]).view(x.size(0), -1))
        reconstructions = reconstructions.view(-1, 1, 28, 28)

        return reconstructions, masked


class CapsNet(nn.Module):
    def __init__(self, opt=None, num_classes=10):
        super(CapsNet, self).__init__()

        self.num_classes = num_classes
        self.conv_layer = ConvLayer()
        self.primary_capsules = PrimaryCaps(opt=opt)
        self.digit_capsules = DigitCaps(num_capsules=num_classes, opt=opt)
        self.decoder = Decoder(opt=opt, num_classes=num_classes)

        self.mse_loss = nn.MSELoss()

    def forward(self, data):
        if isDebug: print(f"CapsNet.forward(data): data.size = {data.size(0)} by {data.size(1)}")
        # conv_data = self.conv_layer(data)
        # U = self.primary_capsules(conv_data)
        # V = self.digit_capsules(U)
        output = self.digit_capsules(self.primary_capsules(self.conv_layer(data)))
        reconstructions, masked = self.decoder(output, data)

        return output, reconstructions, masked

    def loss(self, data, x, target, reconstructions):
        if isDebug: print(f"CapsNet.loss(data, x, target)")
        margin_loss = self.margin_loss(x, target)
        try:
            reconstruction_loss = self.reconstruction_loss(data, reconstructions)
        except Exception as e:
            eprint(f"Error: {e} Couldn't calculate reconstruction loss")
            reconstruction_loss = 0

        return margin_loss + reconstruction_loss

    def margin_loss(self, x, labels, size_average=True):
        batch_size = x.size(0)

        v_c = torch.sqrt((x ** 2).sum(dim=2, keepdim=True))

        left = ((F.relu(0.9 - v_c)) ** 2).view(batch_size, -1)
        right = ((F.relu(v_c - 0.1)) ** 2).view(batch_size, -1)

        loss = labels * left + 0.5 * (1.0 - labels) * right
        loss = loss.sum(dim=1).mean()

        return loss

    def reconstruction_loss(self, data, reconstructions):
        if isDebug: print(f"CapsNet.reconstruction_loss(data, reconstructions): data.size = {data.size(0)} by {data.size(1)}")
        print(f"reconstructions.size() = {reconstructions.size()}")
        r = reconstructions.contiguous().view(reconstructions.size(0), -1)
        d = data.contiguous().view(reconstructions.size(0), -1)

        loss = self.mse_loss(r, d)

        return loss * 0.0005


if __name__ == "__main__":

    capsule_net = CapsNet()
    if USE_CUDA:
        capsule_net = capsule_net.cuda()
    optimizer = Adam(capsule_net.parameters())

    batch_size = BATCH_SIZE  # 128
    mnist = Mnist(batch_size)
    n_epochs = 30

    print("\n-- Train Phase -- \n")
    for epoch in tqdm(range(n_epochs)):
        capsule_net.train()
        train_loss = 0
        for batch_id, (data, target) in enumerate(mnist.train_loader):

            target = torch.sparse.torch.eye(10).index_select(dim=0, index=target)
            data, target = Variable(data), Variable(target)

            if USE_CUDA:
                data, target = data.cuda(), target.cuda()

            # Note that we need to set accumulated gradient from previous batch to 0 before proceeding with backprop
            optimizer.zero_grad()
            output, reconstructions, masked = capsule_net(data)
            loss = capsule_net.loss(data, output, target, reconstructions)
            loss.backward()
            optimizer.step()

            # Warning: This will be an error in PyTorch 0.5. Use tensor.item() to convert a 0-dim tensor to a Python number
            # train_loss += loss.data[0]
            train_loss += loss.data.item()

            if batch_id % BATCH_SIZE == 0:
                train_acc = sum(np.argmax(masked.data.cpu().numpy(), 1) ==
                                np.argmax(target.data.cpu().numpy(), 1)) / float(batch_size)
                print(f"Batch_{batch_id}_train_accuracy = {train_acc}")

        percent_train_loss = train_loss / len(mnist.train_loader)
        print(f"% train loss in epoch{epoch} = {percent_train_loss}")
        # print(train_loss / len(mnist.train_loader))

        print("\n-- Eval Phase -- \n")
        capsule_net.eval()
        test_loss = 0
        for batch_id, (data, target) in enumerate(mnist.test_loader):

            target = torch.sparse.torch.eye(10).index_select(dim=0, index=target)
            data, target = Variable(data), Variable(target)

            if USE_CUDA:
                data, target = data.cuda(), target.cuda()

            output, reconstructions, masked = capsule_net(data)
            loss = capsule_net.loss(data, output, target, reconstructions)

            test_loss += loss.data[0]

            if batch_id % BATCH_SIZE == 0:
                test_acc = sum(np.argmax(masked.data.cpu().numpy(), 1) ==
                               np.argmax(target.data.cpu().numpy(), 1)) / float(batch_size)
                print(f"Batch_{batch_id}_test_accuracy = {test_acc}")

        percent_test_loss = test_loss / len(mnist.test_loader)
        print(f"% test loss in epoch{epoch} = {percent_test_loss}")
        # print(test_loss / len(mnist.test_loader))




# class Mnist:
#     def __init__(self, batch_size):
#         dataset_transform = transforms.Compose([
#             transforms.ToTensor(),
#             transforms.Normalize((0.1307,), (0.3081,))
#         ])
#
#         train_dataset = datasets.MNIST('../data', train=True, download=True, transform=dataset_transform)
#         test_dataset = datasets.MNIST('../data', train=False, download=True, transform=dataset_transform)
#
#         self.train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
#         self.test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
