import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
from torch import index_select

from constants import *

class DilatedConv(nn.Module):
    def __init__(self):
        super(DilatedConv, self).__init__()
        self.conv1 = nn.Conv1d(28, 28, 3, padding=1)
        self.conv2 = nn.Conv1d(28, 28, 3, padding=2, dilation=2)
        self.conv3 = nn.Conv1d(28, 28, 3, padding=4, dilation=4)
        self.conv4 = nn.Conv1d(28, 28, 3, padding=8, dilation=8)
        self.conv5 = nn.Conv1d(28, 28, 3, padding=16, dilation=16)
        self.elu = nn.ELU()
        self.fc = nn.Linear(28, 1)




    def forward(self, input, unpacked_masks):
        x=input

        x = torch.transpose(x, 1, 2)
        x = self.conv1(x)
        x = torch.transpose(x, 1, 2)

        #print("x after conv1", x.data.shape)
        x = torch.mul(x, unpacked_masks)
        x = self.elu(x)

        x = torch.transpose(x, 1, 2)
        x = self.conv2(x)
        x = torch.transpose(x, 1, 2)
        #print("x after conv2", x.data.shape)

        x = torch.mul(x, unpacked_masks)
        x = self.elu(x)

        x = torch.transpose(x, 1, 2)
        x = self.conv3(x)
        x = torch.transpose(x, 1, 2)
        #print("x after conv3", x.data.shape)

        x = torch.mul(x, unpacked_masks)
        x = self.elu(x)

        x = torch.transpose(x, 1, 2)
        x = self.conv4(x)
        x = torch.transpose(x, 1, 2)
        #print("x after conv4", x.data.shape)

        x = torch.mul(x, unpacked_masks)
        x = self.elu(x)

        x = torch.transpose(x, 1, 2)
        x = self.conv5(x)
        x = torch.transpose(x, 1, 2)
        #print("x after conv5", x.data.shape)

        x = torch.mul(x, unpacked_masks)
        x = self.elu(x)

        x = self.fc(x)

        return x