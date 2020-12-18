"""
The file containing the nueral network iteself. Describes the layers, and the forward function. This file is not run by
the user directly, but is invoked by training.py.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicNet(nn.Module):
  """
    A class defining the neural net.

    :param batch_size: An int used to define the batch size of each layer.
    :type batch_size: int
    :param kernel_size: An int the defines the kernel size of the convolutional layers.
    :type kernel_size: int
    :param processor: The number of the GPU to use. The CPU is used if None.
    :type processor: int
    """
    def __init__(self, num_class, batch_size=32, kernel_size=5, processor=None):
        super(BasicNet, self).__init__()
        self.gpu = processorc
        self.bs = batch_size
        self.ks = kernel_size
        self.num_class = num_class

        self.conv1 = nn.Conv2d(3, 6, self.ks)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, self.bs, self.ks)
        print("batch size: " + str(self.bs))
        print("kernel size: " + str(self.ks))

        # fully connected layers
        self.fc1 = nn.Linear(self.bs * int((99 - (self.ks + 1) / 2) ** 2), 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, self.num_class)

    def forward(self, x):
        """
            The forward function of the model.

            :param x: The input to the model as a NumPy Tensor.
            :type x: NumPy Tensor
        """
        # optionnaly moves the model to a GPU
        if self.gpu is not None:
            x.cuda(self.gpu)

        # a mix of pooling and convolutional layers
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        # checks that the layer is the right size, throws an error if not
        x = x.view(x.size(0), self.bs * int((99 - (self.ks + 1) / 2) ** 2))
        # fully connected layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

