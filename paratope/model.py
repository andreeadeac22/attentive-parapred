"""
PyTorch implementation of original Parapred architecture.
"""
from __future__ import print_function
from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

from constants import *

class AbSeqModel(nn.Module):
    def __init__(self):
        """
        Parapred's building blocks.
        """
        super(AbSeqModel, self).__init__()
        # kernel
        self.conv1 = nn.Conv1d(NUM_FEATURES, NUM_FEATURES, 3, padding=1)
        self.elu = nn.ELU()
        self.dropout1 = nn.Dropout(0.15)
        self.bidir_lstm = nn.LSTM(NUM_FEATURES, 256, bidirectional=True)
        self.dropout2 = nn.Dropout(0.3)
        self.fc = nn.Conv1d(512, 1, 1)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        """
        Parameter initialisation for Parapred architecture. Xavier and orthogonal are used, depending on layer.
        :param m:
        :return:
        """
        if isinstance(m, nn.Conv1d):
            torch.nn.init.xavier_uniform(m.weight.data)
            m.bias.data.fill_(0.0)
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
        """
        Performing forward propagation
        :param input: antibody amino acid sequences
        :param unpacked_masks:
        :param masks:
        :param lengths:
        :return: binding probabilities for antibody amino acids.
        """
        initial = input
        x = input

        x = torch.transpose(x, 1, 2)
        x = self.conv1(x)
        x = torch.transpose(x, 1, 2)

        x = torch.mul(x, unpacked_masks)
        x = self.elu(x)

        x = x + initial

        x = self.dropout1(x)

        packed_input = pack_padded_sequence(x, lengths, batch_first=True)
        output, hidden = self.bidir_lstm(packed_input)
        x, _ = pad_packed_sequence(output, batch_first=True)
        print("after lstm", x.data, file=print_file)  # 32x32x512

        x = self.dropout2(x)

        x = torch.transpose(x, 1, 2)
        x = self.fc(x)
        x = torch.transpose(x, 1, 2)

        x = torch.mul(x, masks)
        return x
