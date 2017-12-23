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
        self.bidir_lstm = nn.LSTM(28, 256, bidirectional = True)
        self.dropout2 = nn.Dropout(0.3)
        self.conv2 = nn.Conv1d(512, 1, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input, lengths):
        initial = input
        x = input

        #print("initial x", x.data.shape)

        x = torch.transpose(x, 1, 2)

        #print("before conv1 x", x.data.shape)

        # Conv1D
        # elu
        # l2 regularizer(in optimizer)
        x = self.conv1(x)

        #print("after conv1", x.data.shape)

        x = torch.transpose(x, 1, 2)
        x = self.elu(x)

        #print("after elu", x.data.shape)

        # multiply x with the mask

        x = self.dropout1(x)

        #Add residual connections
        #print("after dropout", x.data.shape)
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
        #print("after lstm", x.data.shape)


        #Dropout
        x = self.dropout2(x)
        #print("after dropout 2", x.data.shape)

        x = torch.transpose(x, 1, 2)

        # Time-distributed?
        # dense
        # sigmoid
        x = self.conv2(x)
        #print("after dense - linear = conv2", x.data.shape)

        #x = self.fully(x)

        x = torch.transpose(x, 1, 2)

        #print("after linear", x.data.shape)
        x = self.sigmoid(x)
        #print("at the end", x.data.shape)
        return x