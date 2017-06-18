import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
import torchvision.models as models

model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
}

class SegmentationNetwork(nn.Module):

    def __init__(self):
        super(SegmentationNetwork, self).__init__()

        ############################################################################
        #                             YOUR CODE                                    #
        ############################################################################
        self.alexNet = nn.Sequential(
            models.alexnet(pretrained = True).features
            #list(models.alexnet(pretrained = True).features)[:-6]
        )
        '''
            output = (input - filter + 2*padding)/stride + 1
            In convTranspose input and output are exchanged
        '''
        self.fcn = nn.Sequential(
            nn.Dropout(),
            #256,6,6
            nn.Conv2d(256, 4096, kernel_size = 6, stride = 1),
            #4096,1,1
            nn.ReLU(inplace = True),
            nn.Dropout(),
            nn.Conv2d(4096, 4096, kernel_size = 1, stride = 1),
            #4096,1,1
            nn.ReLU(inplace = True),
            nn.Conv2d(4096, 23, kernel_size = 1, stride = 1),
            #23,1,1
        )

        self.upsample = nn.Sequential(
            #23,1,1
            nn.ConvTranspose2d(23, 23, kernel_size = 240, stride = 32),
            # 23,240,240
            nn.ReLU(inplace = True)
        )
        '''
            output = strides(input - 1) + filter - 2*padding
            It is just the opposite operation of convolution
        self.upsample = nn.Sequential(
            # input = 256,6,6
            nn.ConvTranspose2d(256, 128, kernel_size = 2, stride = 2),
            # 128,12,12
            nn.BatchNorm2d(128),
            nn.ReLU(inplace = True),
            nn.ConvTranspose2d(128, 64, kernel_size = 4, stride = 2),
            # 64,26,26
            nn.BatchNorm2d(64),
            nn.ReLU(inplace = True),
            nn.ConvTranspose2d(64, 32, kernel_size = 4, stride = 3),
            # 32,79,79
            nn.BatchNorm2d(32),
            nn.ReLU(inplace = True),
            nn.ConvTranspose2d(32, 23, kernel_size = 6, stride = 3),
            # 23,240,240
            nn.ReLU(inplace = True)
        )
        '''


    def forward(self, x):
        """
        Forward pass of the convolutional neural network. Should not be called
        manually but by calling a model instance directly.

        Inputs:
        - x: PyTorch input Variable
        """
        ############################################################################
        #                             YOUR CODE                                    #
        ############################################################################
        out = self.alexNet(x)
        out = self.fcn(out)
        out = self.upsample(out)
        #out = out.view(0,2,3,1)
        #out = out.transpose(1,2).transpose(2,3).contiguous()
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

    def init_alexnet_weight(self):
        # output = 256,6,6
        self.alexNet = models.alexnet(pretrained = True)
        #self.alexNet.load_state_dict(model_zoo.load_url(model_urls['alexnet']))

    """
    def parameters(self):
        return self.upsample.parameters()
    """