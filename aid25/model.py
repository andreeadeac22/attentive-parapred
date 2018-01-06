from __future__ import print_function
from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

from preprocessing import NUM_FEATURES
from constants import *

class AbSeqModel(nn.Module):
    def __init__(self):
        super(AbSeqModel, self).__init__()
        # kernel
        self.conv1 = nn.Conv1d(28, 28, 3, padding = 1)
        self.elu = nn.ELU()
        self.dropout1 = nn.Dropout(0.15)
        self.bidir_lstm = nn.LSTM(28, 256, bidirectional = True, dropout=0.15)
        self.dropout2 = nn.Dropout(0.3)
        self.conv2 = nn.Conv1d(512, 1, 1)
        self.sigmoid = nn.Sigmoid()

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self,m):
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
        initial = input
        x = input

        #print("initial x", x.data, file=track_f)

        x = torch.transpose(x, 1, 2)

        #print("before conv1 x", x.data, file=track_f)

        # Conv1D
        # elu
        # l2 regularizer(in optimizer)
        x = self.conv1(x)

        #print("after conv1", x.data, file=print_file)

        x = torch.transpose(x, 1, 2)

        #print("before mul 1", x.data, file=print_file)

        x = torch.mul(x, unpacked_masks)

        x = self.elu(x)

        print("after elu", x.data, file=print_file)

        # multiply x with the mask

        #x = self.dropout1(x)

        #Add residual connections
        #print("after dropout", x.data, file=print_file)
        #print("initial", initial.data.shape)
        x = x + initial


        # Bidirectional LSTM
        # dropout
        # recurrent dropout
        # -- need to batch - edgar's doing 32.
        # probably need to sort and batch sequences with equal number of residues

        # x is 1434x32x28
        # make 1434 matrices of the form 32x28 and sort (descending) based on how many residues (out of 32) are present (residue.feature != 0's).
        # Result: 1434 tuples, where i'th tuple = ( matrix 32x28, length (used for sorting), position in initial matrix 1434x32x28)

        # then for each batch (of size 32),
        #   packed = pack_padded_sequence(x, lengths, batch_first = True) - where x is 32x31x28
        #   output, hidden = self.bidir_lstm(packed)
        #   x, _ = pad_packed_sequence(output, batch_first = True)

        # should i then redo the order of the matrix 1434x...512x1? using the positions from the initial matrix?


        packed_input = pack_padded_sequence(x, lengths, batch_first=True)

        output, hidden = self.bidir_lstm(packed_input)

        x, _ = pad_packed_sequence(output, batch_first = True)
        print("after lstm", x.data, file=print_file)


        #Dropout
        x = self.dropout2(x)
        #print("after dropout 2", x.data, file=print_file)

        x = torch.transpose(x, 1, 2)

        # Time-distributed?
        # dense
        # sigmoid
        x = self.conv2(x)
        #print("after dense - linear = conv2", x.data, file=track_f)

        x = torch.transpose(x, 1, 2)

        print("after conv2", x.data, file=print_file)
        #print("after sigmoid", x.data, file=print_file)

        x = torch.mul(x, masks)
        #print("after multi", x.data, file=track_f)


        #print("x at the end", x, file=track_f)

        return x