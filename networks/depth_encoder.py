from __future__ import absolute_import, division, print_function

import numpy as np

import torch
import torch.nn as nn
import torchvision.models as models
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
import math
from .mpvit import *


class DepthNet(nn.Module):
    def __init__(self):
        super(DepthNet, self).__init__()
        self.num_ch_enc = np.array([64, 96, 176, 216, 216])
        self.encoder=mpvit_tiny()
    

    def forward(self, input_image):
        self.features = []
        x = (input_image - 0.45) / 0.225
        self.features = self.encoder(x)

        return self.features
        
