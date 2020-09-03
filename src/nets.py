import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.utils.data as data
import torchvision
from torchvision import transforms
import torch.optim as optim
import torch.multiprocessing
import logging
from datetime import datetime, date, time


class BasicNet(nn.Module):
    def __init__(self, batch_size=32, kernel_size=5, processor=None):
        super(BasicNet, self).__init__()
        self.gpu = processor
        self.bs = batch_size
        self.ks = kernel_size
        self.num_class = 10
        self.conv1 = nn.Conv2d(3, 6, self.ks)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, self.bs, self.ks)
        print("batch size: " + str(self.bs))
        print("kernel size: " + str(self.ks))
        self.fc1 = nn.Linear(self.bs * int((99 - (self.ks + 1) / 2) ** 2), 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, self.num_class)

    def forward(self, x):
        if self.gpu is not None:
            x.cuda(self.gpu)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), self.bs * int((99 - (self.ks + 1) / 2) ** 2))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        if self.gpu is not None:
            output = x.detach()
            del x
            return output
        return x


# class Resnet18Grayscale(ResNet):
class Resnet18Grayscale(nn.Module):
    '''
    A Resnet18 variant that accepts single-channel
    grayscale images instead of RGB.

    Using this class saves space from not having
    to replicate our single-layer spectrograms three
    times to pretend they are RGB images.
    '''

    # ------------------------------------
    # Constructor
    # -------------------

    def __init__(self, *args, **kwargs):
        '''
        Args and kwargs as per https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
        class ResNet.__init__()

        '''
        # The [2,2,2,2] is an instruction to the
        # superclass' __init__() for how many layers
        # of each type to create. This info makes the
        # ResNet into a ResNet18:
        self.num_class = 10
        super().__init__(BasicBlock, [2, 2, 2, 2], *args, **kwargs)

        # Change expected channels from 3 to 1
        # The superclass created first layer
        # with the first argument being a 3.
        # We just replace the first layer:
        self.inplanes = 64  # ******* Should be batch size?
        self.conv1 = nn.Conv2d(1, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)

    # ------------------------------------
    # forward
    # -------------------

    def forward(self, x):
        out_logit = super().forward(x)

        # Since we have binary classification,
        # the Sigmoid function does what a
        # softmax would do for multi-class:

        out_probs = nn.Sigmoid()(out_logit)
        return out_probs

    # ------------------------------------
    # device
    # -------------------

    def device(self):
        '''
        Returns device where model resides.
        Can use like this to move a tensor
        to wherever the model is:

            some_tensor.to(<model_instance>.device())

        '''
        return next(self.parameters()).device
