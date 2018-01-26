import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
from torch import index_select

from constants import *
from preprocessing import NUM_FEATURES

class DilatedConv(nn.Module):
    def __init__(self):
        super(DilatedConv, self).__init__()
        self.conv1 = nn.Conv1d(NUM_FEATURES, 32, 3, padding=1)
        self.conv2 = nn.Conv1d(32, 64, 3, padding=2, dilation=2)
        self.conv3 = nn.Conv1d(64, 128, 3, padding=4, dilation=4)
        self.conv4 = nn.Conv1d(128, 256, 3, padding=8, dilation=8)
        self.conv5 = nn.Conv1d(256, 256, 3, padding=16, dilation=16)
        self.conv6 = nn.Conv1d(256, 512, 3, padding=32, dilation=32)
        self.elu = nn.ELU()
        self.dropout = nn.Dropout()
        self.fc = nn.Linear(512, 1, 1)


        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Conv1d):
            torch.nn.init.xavier_uniform(m.weight.data)
            m.bias.data.fill_(0.0)

    def forward(self, input, unpacked_masks):
        x=input

        unpacked_masks = torch.transpose(unpacked_masks, 1, 2)

        x = torch.transpose(x, 1, 2)
        x = self.conv1(x)

        #print("x after conv1", x.data.shape)
        x = torch.mul(x, unpacked_masks)
        x = self.elu(x)

        x = self.conv2(x)
        #print("x after conv2", x.data.shape)

        x = torch.mul(x, unpacked_masks)
        x = self.elu(x)

        x = self.conv3(x)
        #print("x after conv3", x.data.shape)

        x = torch.mul(x, unpacked_masks)
        x = self.elu(x)

        x = self.conv4(x)
        #print("x after conv4", x.data.shape)

        x = torch.mul(x, unpacked_masks)
        x = self.elu(x)

        x = self.conv5(x)
        #print("x after conv5", x.data.shape)

        x = torch.mul(x, unpacked_masks)
        x = self.elu(x)

        # print("x after conv5", x.data.shape)

        x = self.conv6(x)
        # print("x after conv5", x.data.shape)

        x = torch.mul(x, unpacked_masks)
        x = self.elu(x)

        x = torch.transpose(x, 1, 2)

        x = self.dropout(x)

        x = self.fc(x)

        #print("x after fc", x.data.shape)

        return x