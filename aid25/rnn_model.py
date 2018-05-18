"""
RNN Baseline.
"""
from __future__ import print_function
from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

from .constants import *


class RNNModel(nn.Module):
    def __init__(self):
        """
        Building blocks of the baseline -  LSTM layer and dense output layer
        """
        super(RNNModel, self).__init__()
        self.bidir_lstm = nn.LSTM(NUM_FEATURES, 256)
        self.fc = nn.Conv1d(256, 1, 1)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.LSTM):
            torch.nn.init.xavier_uniform(m.weight_ih_l0)
            torch.nn.init.orthogonal(m.weight_hh_l0)
            for names in m._all_weights:
                for name in filter(lambda n: "bias" in n, names):
                    bias = getattr(m, name)
                    n = bias.size(0)
                    start, end = n // 4, n // 2
                    bias.data[start:end].fill_(1.0)
            m.bias_ih_l0.data.fill_(0.0)

    def forward(self, input, unpacked_masks, masks, lengths):
        x = input

        packed_input = pack_padded_sequence(x, lengths, batch_first=True)
        output, hidden = self.bidir_lstm(packed_input)
        x, _ = pad_packed_sequence(output, batch_first=True)
        print("after lstm", x.data, file=print_file)  # 32x32x512

        x = torch.transpose(x, 1, 2)
        x = self.fc(x)
        x = torch.transpose(x, 1, 2)

        x = torch.mul(x, masks)
        return x
