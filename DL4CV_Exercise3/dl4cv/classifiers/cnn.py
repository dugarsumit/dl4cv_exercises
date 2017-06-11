import os

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class ThreeLayerCNN(nn.Module):
    """
    A PyTorch implementation of a three-layer convolutional network
    with the following architecture:

    conv - relu - 2x2 max pool - fc - dropout - relu - fc

    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """

    def __init__(self, input_dim=(3, 32, 32), num_filters=32, kernel_size=7,
                 stride=1, weight_scale=0.001, pool=2, hidden_dim=100,
                 num_classes=10, dropout=0.0):
        """
        Initialize a new network.

        Inputs:
        - input_dim: Tuple (C, H, W) giving size of input data.
        - num_filters: Number of filters to use in the convolutional layer.
        - filter_size: Size of filters to use in the convolutional layer.
        - hidden_dim: Number of units to use in the fully-connected hidden layer-
        - num_classes: Number of scores to produce from the final affine layer.
        - stride: The size of the window to take a max over.
        - weight_scale: Scale for the convolution weights initialization-
        - pool: The size of the max pooling window.
        - dropout: Probability of an element to be zeroed.
        """
        super(ThreeLayerCNN, self).__init__()
        channels, height, width = input_dim

        ############################################################################
        # TODO: Initialize the necessary layers to resemble the ThreeLayerCNN      #
        # architecture  from the class docstring. In- and output features should   #
        # not be hard coded which demands some calculations especially for the     #
        # input of the first fully convolutional layer. The convolution should use #
        # "same" padding which can be derived from the kernel size and its weights #
        # should be scaled. Layers should have a bias if possible.                 #
        ############################################################################
        out_height = height
        pad = ((out_height - 1)*stride - height + kernel_size)/2
        input_l2 = num_filters*(height/pool)*(width/pool)
        input_l3 = hidden_dim
        conv_weights = nn.Parameter(
            weight_scale*torch.randn(num_filters, channels, kernel_size, kernel_size))
        self.l1 = nn.Sequential()
        self.conv = nn.Conv2d(in_channels = channels,
                         out_channels = num_filters,
                         kernel_size = kernel_size,
                         stride = stride,
                         padding = pad)
        #self.conv.weight = nn.Parameter(weight_scale * self.conv.weight.data)
        self.l1.add_module("conv", self.conv)
        #self.l1.add_module("bn", nn.BatchNorm2d(num_filters))
        self.l1.add_module("relu", nn.ReLU())
        self.l1.add_module("max_pool", nn.MaxPool2d(kernel_size = pool))
        self.l2 = nn.Sequential(
            nn.Linear(in_features = input_l2,
                      out_features = hidden_dim),
            nn.Dropout(p = dropout),
            nn.ReLU()
        )
        self.l3 = nn.Sequential(
            nn.Linear(in_features = input_l3,
                      out_features = num_classes)
        )
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

    def forward(self, x):
        """
        Forward pass of the convolutional neural network. Should not be called
        manually but by calling a model instance directly.

        Inputs:
        - x: PyTorch input Variable
        """

        ############################################################################
        # TODO: Chain our previously initialized convolutional neural network      #
        # layers to resemble the architecture drafted in the class docstring.      #
        # Have a look at the Variable.view function to make the transition from    #
        # convolutional to fully connected layers.                                 #
        ############################################################################
        out_l1 = self.l1(x)
        out_l1 = out_l1.view(out_l1.size(0), -1)
        out_l2 = self.l2(out_l1)
        out = self.l3(out_l2)
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return out

    def save(self, path):
        """
        Save model with its parameters to the given path. Conventionally the
        path should end with "*.model".

        Inputs:
        - path: path string
        """
        print 'Saving model... %s' % path
        torch.save(self, path)

    def load(self, path):
        return torch.load(map_location = path)
