import torch
import torch.nn as nn
from torch.autograd import Function, Variable

from functools                              import reduce
from torch.legacy.nn.Module                 import Module as LegacyModule
from torch.legacy.nn.utils                  import clear
from torch.nn._functions.thnn.normalization import CrossMapLRN2d

import numpy as np


class render4cnn(nn.Module):
    def __init__(self, finetune=False, weights = None, num_classes = 12):
        super(render4cnn, self).__init__()

        # Normalization layers
        norm1 = nn.LocalResponseNorm(5, 0.0001, 0.75, 1)
        norm2 = nn.LocalResponseNorm(5, 0.0001, 0.75, 1)

        # conv layers
        conv1 = nn.Conv2d(3, 96, (11, 11), (4,4))
        relu1 = nn.ReLU()
        pool1 = nn.MaxPool2d( (3,3), (2,2), (0,0), ceil_mode=True)

        conv2 = nn.Conv2d(96, 256, (5, 5), (1,1), (2,2), 1,2)
        relu2 = nn.ReLU()
        pool2 = nn.MaxPool2d( (3,3), (2,2), (0,0), ceil_mode=True)

        conv3 = nn.Conv2d(256,384,(3, 3),(1, 1),(1, 1))
        relu3 = nn.ReLU()

        conv4 = nn.Conv2d(384,384,(3, 3),(1, 1),(1, 1),1,2)
        relu4 = nn.ReLU()

        conv5 = nn.Conv2d(384,256,(3, 3),(1, 1),(1, 1),1,2)
        relu5 = nn.ReLU()
        pool5 = nn.MaxPool2d((3, 3),(2, 2),(0, 0),ceil_mode=True)


        # inference layers
        fc6     = nn.Linear(9216,4096)
        relu6   = nn.ReLU()

        fc7     = nn.Linear(4096,4096)
        relu7   = nn.ReLU()

        drop6 = nn.Dropout(0.5)
        drop7 = nn.Dropout(0.5)

        azim     = nn.Linear(4096,num_classes * 360)
        elev     = nn.Linear(4096,num_classes * 360)
        tilt     = nn.Linear(4096,num_classes * 360)

        # Define Network
        self.conv4 = nn.Sequential( conv1, relu1, pool1, norm1,
                                    conv2, relu2, pool2, norm2,
                                    conv3, relu3,
                                    conv4, relu4)

        self.conv5 = nn.Sequential( conv5,  relu5,  pool5)

        self.infer = nn.Sequential( fc6,    relu6,  drop6,
                                    fc7,    relu7,  drop7)

        if finetune:
            self.conv4.requires_grad = False
            self.conv5.requires_grad = False


        self.azim = nn.Sequential( azim )
        self.elev = nn.Sequential( elev )
        self.tilt = nn.Sequential( tilt )

    def forward(self, images):
        features = self.conv4(images)
        features = self.conv5(features)
        features = features.view(features.size(0), 9216)
        features = self.infer(features)

        azim = self.azim(features)
        elev = self.elev(features)
        tilt = self.tilt(features)

        return azim, elev, tilt
