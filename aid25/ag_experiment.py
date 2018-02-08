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

from constants import *
from preprocessing import NUM_FEATURES, AG_NUM_FEATURES

class AG(nn.Module):
    def __init__(self):
        super(AG, self).__init__()
        self.conv1 = nn.Conv1d(NUM_FEATURES, 64, 3, padding=1)

        self.agconv1 = nn.Conv1d(AG_NUM_FEATURES, 64, 3, padding=1)

        self.conv2 = nn.Conv1d(64, 128, 3, padding=2, dilation=2)

        self.agconv2 = nn.Conv1d(64, 128, 3, padding=2, dilation=2)

        self.conv3 = nn.Conv1d(128, 256, 3, padding=4, dilation=4)

        self.agconv3 = nn.Conv1d(128, 256, 3, padding=4, dilation=4)

        #self.conv_fix = nn.Conv1d(1269, 32, 3, padding=1)

        self.bn1 = nn.BatchNorm1d(64)
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

        self.maxpool1 = nn.MaxPool1d(2)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Conv1d):
            torch.nn.init.xavier_uniform(m.weight.data)
            m.bias.data.fill_(0.0)
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform(m.weight.data)
            m.bias.data.fill_(0.0)

    def forward(self, ab_input, ab_unpacked_masks, ag_input, ag_unpacked_masks):
        x=ab_input
        agx = ag_input

        ab_unpacked_masks = torch.transpose(ab_unpacked_masks, 1, 2)
        ag_unpacked_masks = torch.transpose(ag_unpacked_masks, 1, 2)

        x = torch.transpose(x, 1, 2)
        agx = torch.transpose(agx, 1, 2)

        x = self.conv1(x)
        agx = self.agconv1(agx)

        #print("x after conv1", x.data.shape)
        x = torch.mul(x, ab_unpacked_masks)
        x = self.bn1(x)
        x = self.elu(x)
        x = torch.mul(x, ab_unpacked_masks)
        x = self.dropout(x)

        agx = torch.mul(agx, ag_unpacked_masks)
        agx = self.bn1(agx)
        agx = self.elu(agx)
        agx = torch.mul(agx, ag_unpacked_masks)
        agx = self.dropout(agx)

        # MaxPool
        # print("agx after conv1", agx.shape)
        agx = self.maxpool1(agx)
        # print("agx after pool1", agx.shape)
        ag_unpacked_masks = self.maxpool1(ag_unpacked_masks)

        x = self.conv2(x)
        #print("x after conv2", x.data.shape)

        x = torch.mul(x, ab_unpacked_masks)
        x = self.bn2(x)
        x = self.elu(x)
        x = torch.mul(x, ab_unpacked_masks)

        x = self.dropout(x)


        agx = self.agconv2(agx)
        # print("x after conv2", x.data.shape)

        agx = torch.mul(agx, ag_unpacked_masks)
        agx = self.bn2(agx)
        agx = self.elu(agx)
        agx = torch.mul(agx, ag_unpacked_masks)

        agx = self.dropout(agx)

        # MaxPool
        # print("agx after conv1", agx.shape)
        agx = self.maxpool1(agx)
        # print("agx after pool1", agx.shape)
        ag_unpacked_masks = self.maxpool1(ag_unpacked_masks)


        x = self.conv3(x)
        #print("x after conv3", x.data.shape)

        x = torch.mul(x, ab_unpacked_masks)
        x = self.bn3(x)
        x = self.elu(x)
        x = torch.mul(x, ab_unpacked_masks)

        x = self.dropout(x)

        agx = self.agconv3(agx)
        # print("x after conv3", x.data.shape)

        agx = torch.mul(agx, ag_unpacked_masks)
        agx = self.bn3(agx)
        agx = self.elu(agx)
        agx = torch.mul(agx, ag_unpacked_masks)

        agx = self.dropout(agx)

        # MaxPool
        # print("agx after conv1", agx.shape)
        agx = self.maxpool1(agx)
        # print("agx after pool1", agx.shape)
        ag_unpacked_masks = self.maxpool1(ag_unpacked_masks)

        old = x

        oldag = agx

        #print("x shape", x.shape)
        #print("agx shape", agx.shape)

        w_1 = self.aconv1(x)
        w_2 = self.aconv2(agx)
        #w_2 = self.aconv2(x)

        #print("w_1", w_1.shape)
        #print("w_2", w_2.shape)

        w = self.lrelu(w_2 + torch.transpose(w_1, 1, 2))

        #print("w", w.shape)

        bias_mat = 1e9 * (ag_unpacked_masks - 1.0)

        #print("bias_mat", bias_mat.shape)

        w = self.softmax(w + bias_mat)
        #print("w", w.data.cpu().numpy(), file=track_f)

        #print("shape", torch.transpose(agx, 1, 2).shape)
        x = torch.bmm(w, torch.transpose(agx, 1, 2))

       # x = torch.bmm(w, torch.transpose(x,1,2))

        #print("after bmm", x.shape)

        x = torch.transpose(x, 1, 2)

        x = x + old
        x = torch.cat((x, old), dim=1)
        x = torch.mul(x, ab_unpacked_masks)

        x = self.bn4(x)
        x = self.elu(x)
        x = torch.mul(x, ab_unpacked_masks)
        x = torch.transpose(x, 1, 2)

        x = self.dropout2(x)

        x = self.fc(x)

        #print("x after fc", x.data.shape)

        return x