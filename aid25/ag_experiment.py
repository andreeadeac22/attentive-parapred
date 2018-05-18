"""
AG-Fast-Parapred Model
"""
from __future__ import print_function
from sklearn.model_selection import KFold

import numpy as np
np.set_printoptions(threshold=np.nan)
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
from torch import index_select

from .constants import *
from .preprocessing import NUM_FEATURES, AG_NUM_FEATURES

class AG(nn.Module):
    def __init__(self):
        """
        Declares the building blocks of the neural network.
        """
        super(AG, self).__init__()

        self.conv1 = nn.Conv1d(NUM_FEATURES, 64, 3, padding=1)  # antibody first a trous convolutional layer

        self.agconv1 = nn.Conv1d(AG_NUM_FEATURES, 64, 3, padding=1)   # antigen first a trous convolutional layer

        self.conv2 = nn.Conv1d(64, 128, 3, padding=2, dilation=2)  #antibody second a trous convolutional layer

        self.agconv2 = nn.Conv1d(64, 128, 3, padding=2, dilation=2)  #antigen second a trous convolutional layer

        self.conv3 = nn.Conv1d(128, 256, 3, padding=4, dilation=4)  #antibody third a trous convolutional layer

        self.agconv3 = nn.Conv1d(128, 256, 3, padding=4, dilation=4)

        #self.agconv4 = nn.Conv1d(256, 32, 1)


        self.bn1 = nn.BatchNorm1d(64)  # batch normalisation after the first convolutional layer for antibody
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)
        self.bn4 = nn.BatchNorm1d(512)

        self.agbn1 = nn.BatchNorm1d(64)  # batch normalisation after the first convolutional layer for antigen
        self.agbn2 = nn.BatchNorm1d(128)
        self.agbn3 = nn.BatchNorm1d(256)

        self.elu = nn.ReLU()
        self.dropout = nn.Dropout(0.15)
        self.dropout2 = nn.Dropout(0.5)
        self.fc = nn.Linear(512, 1, 1)  # dense prediction layer
        self.softmax = nn.Softmax(dim=-1)
        self.lrelu = nn.LeakyReLU(0.2)

        self.aconv1 = nn.Conv1d(256, 1, 1)  # attentional mechanism
        self.aconv2 = nn.Conv1d(32, 1, 1)


        self.maxpool1 = nn.MaxPool1d(2)
        self.maxpool2 = nn.MaxPool1d(4)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Conv1d):
            torch.nn.init.xavier_uniform(m.weight.data)
            m.bias.data.fill_(0.0)
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform(m.weight.data)
            m.bias.data.fill_(0.0)

    def forward(self, ab_input, ab_unpacked_masks, ag_input, ag_unpacked_masks, dist):
        """
        Forward propagation step
        :param ab_input: antibody amino acid sequences
        :param ab_unpacked_masks: antibody amino acid masks
        :param ag_input: antigen amino acid sequences
        :param ag_unpacked_masks: antigen amino acid masks
        :param dist: maskin
        :return: antibody binding probabilities, attentional coefficients
        """
        x=ab_input
        agx = ag_input

        ab_unpacked_masks = torch.transpose(ab_unpacked_masks, 1, 2)
        ag_unpacked_masks = torch.transpose(ag_unpacked_masks, 1, 2)

        x = torch.transpose(x, 1, 2)
        agx = torch.transpose(agx, 1, 2)

        x = self.conv1(x)
        x = torch.mul(x, ab_unpacked_masks)
        x = self.bn1(x)
        x = self.elu(x)
        x = torch.mul(x, ab_unpacked_masks)
        x = self.dropout(x)

        agx = self.agconv1(agx)
        agx = torch.mul(agx, ag_unpacked_masks)
        agx = self.agbn1(agx)
        agx = self.elu(agx)
        agx = torch.mul(agx, ag_unpacked_masks)
        agx = self.dropout(agx)

        x = self.conv2(x)
        x = torch.mul(x, ab_unpacked_masks)
        x = self.bn2(x)
        x = self.elu(x)
        x = torch.mul(x, ab_unpacked_masks)
        x = self.dropout(x)

        #agx = self.maxpool1(agx)
        #ag_unpacked_masks = self.maxpool1(ag_unpacked_masks)

        agx = self.conv2(agx)
        agx = torch.mul(agx, ag_unpacked_masks)
        agx = self.agbn2(agx)
        agx = self.elu(agx)
        agx = torch.mul(agx, ag_unpacked_masks)
        agx = self.dropout(agx)


        x = self.conv3(x)
        x = torch.mul(x, ab_unpacked_masks)
        x = self.bn3(x)
        x = self.elu(x)
        x = torch.mul(x, ab_unpacked_masks)
        x = self.dropout(x)

        #agx = self.maxpool1(agx)
        #ag_unpacked_masks = self.maxpool1(ag_unpacked_masks)

        agx = self.conv3(agx)
        agx = torch.mul(agx, ag_unpacked_masks)
        agx = self.agbn3(agx)
        agx = self.elu(agx)
        agx = torch.mul(agx, ag_unpacked_masks)
        agx = self.dropout(agx)

        # MaxPool
        #agx = self.maxpool1(agx)
        #ag_unpacked_masks = self.maxpool1(ag_unpacked_masks)

        old = x

        oldag = agx

        heads_no = 1
        #bias_mat = 1e9 * (ag_unpacked_masks - 1.0)
        dist_mat = 1e9 * (dist - 1.0)
        #dist_mat = self.maxpool1(dist_mat)

        for i in range(heads_no):
            #agconvi = nn.Conv1d(256, 128, 1)
            aconvi1 = nn.Conv1d(256, 1, 1)
            aconvi2 = nn.Conv1d(256, 1, 1)
            if use_cuda:
                aconvi1.cuda()
                aconvi2.cuda()
                #agconvi.cuda()
            #agx = agconvi(oldag)
            w_1 = aconvi1(x)
            w_2 = aconvi2(agx)
            w = self.lrelu(w_2 + torch.transpose(w_1, 1, 2))
            w = self.softmax(w + dist_mat)
            w = self.dropout(w)
            temp_loop_x = torch.bmm(w, torch.transpose(agx, 1, 2))
            if i==0:
                loop_x = temp_loop_x
            else:
                loop_x = torch.cat((loop_x, temp_loop_x), dim=2)

        x = torch.transpose(loop_x, 1, 2)
        #x = x + old
        x = torch.cat((x, old), dim=1)
        x = torch.mul(x, ab_unpacked_masks)

        x = self.bn4(x)
        x = self.elu(x)
        x = torch.mul(x, ab_unpacked_masks)
        x = torch.transpose(x, 1, 2)

        x = self.dropout2(x)

        x = self.fc(x)

        #print("x after fc", x, file=track_f)

        return x, w