"""
Fast-Parapred Model
"""
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
from torch import index_select

from .constants import *

class AtrousSelf(nn.Module):
    def __init__(self):
        super(AtrousSelf, self).__init__()
        self.conv1 = nn.Conv1d(NUM_FEATURES, 64, 3, padding=1)  # a trous convolution
        self.conv2 = nn.Conv1d(64, 128, 3, padding=2, dilation=2)
        self.conv3 = nn.Conv1d(128, 256, 3, padding=4, dilation=4)
        self.bn1 = nn.BatchNorm1d(64)  # batch normalisation after first a trous convolution
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)
        self.bn4 = nn.BatchNorm1d(512)
        self.elu = nn.ReLU()
        self.dropout = nn.Dropout(0.15)
        self.dropout2 = nn.Dropout(0.5)
        self.fc = nn.Linear(512, 1, 1)
        self.softmax = nn.Softmax(dim=-1)
        self.lrelu = nn.LeakyReLU(0.2)

        self.aconv1 = nn.Conv1d(256, 1, 1)
        self.aconv2 = nn.Conv1d(256, 1, 1)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        """

        :param m: Layer type
        :return: void. Performs parameter initialisation
        """
        if isinstance(m, nn.Conv1d):
            torch.nn.init.xavier_uniform(m.weight.data)
            m.bias.data.fill_(0.0)
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform(m.weight.data)
            m.bias.data.fill_(0.0)

    def forward(self, input, unpacked_masks):
        """

        :param input: antibody amino acid sequences
        :param unpacked_masks:
        :return: antibody binding probabilities
        """
        x=input

        unpacked_masks = torch.transpose(unpacked_masks, 1, 2)

        x = torch.transpose(x, 1, 2)
        x = self.conv1(x)

        #print("x after conv1", x.data.shape)
        x = torch.mul(x, unpacked_masks)
        x = self.bn1(x)
        x = self.elu(x)
        x = torch.mul(x, unpacked_masks)
        x = self.dropout(x)

        x = self.conv2(x)
        #print("x after conv2", x.data.shape)

        x = torch.mul(x, unpacked_masks)
        x = self.bn2(x)
        x = self.elu(x)
        x = torch.mul(x, unpacked_masks)

        x = self.dropout(x)

        x = self.conv3(x)
        #print("x after conv3", x.data.shape)

        x = torch.mul(x, unpacked_masks)
        x = self.bn3(x)
        x = self.elu(x)
        x = torch.mul(x, unpacked_masks)

        x = self.dropout(x)

        old = x

        #print("x", x.data.shape)

        w_1 = self.aconv1(x)
        #print("w_1", w_1.data.shape)

        w_2 = self.aconv2(x)
        #print("w_2", w_2.data.shape)


        w = self.lrelu(w_1 + torch.transpose(w_2, 1, 2))

        #print("w", w.data.shape)

        bias_mat = 1e9 * (unpacked_masks - 1.0)
        w = self.softmax(w + bias_mat)

        x = torch.bmm(w, torch.transpose(x, 1, 2))
        #print("x after bmm", x.data.shape)

        x = torch.transpose(x, 1, 2)

        #x = x + old
        x = torch.cat((x, old), dim=1)
        x = torch.mul(x, unpacked_masks)

        x = self.bn4(x)
        x = self.elu(x)
        x = torch.mul(x, unpacked_masks)
        x = torch.transpose(x, 1, 2)

        x = self.dropout2(x)

        x = self.fc(x)

        #print("x after fc", x.data.shape)

        return x