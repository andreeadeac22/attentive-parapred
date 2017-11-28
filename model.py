from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from preprocessing import NUM_FEATURES

class AbSeqModel(nn.Module):
    def __init__(self):
        super(AbSeqModel, self).__init__()
        # kernel
        self.conv1 = nn.Conv1d(32, 28, 3, padding = (1,1))
        self.dropout1 = nn.Dropout(0.15)
        self.bidir_lstm = nn.LSTM(256, 256, bidirectional = True)
        self.dropout = nn.Dropout(0.3)
        self.fully = nn.Linear(512, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        initial = x

        # Conv1D
        # elu
        # l2 regularizer(in optimizer)
        x = F.elu(self.conv1(x))

        x = self.dropout1(x)

        #Add residual connections
        print("x", x.data.shape)
        print("initial", initial.data.shape)
        x = x + initial

        # Bidirectional LSTM
        # dropout
        # recurrent dropout
        # -- need to batch - edgar's doing 32.
        # probably need to sort and batch sequences with equal number of residues

        packed = pack_padded_sequence(x, lengths, batch_first = True)

        output, hidden = self.bidir_lstm(packed)

        x, _ = pad_packed_sequence(output, batch_first = True)

        #Dropout
        x = self.dropout2(x)

        # Time-distributed?
        # dense
        # sigmoid
        x = self.fully(x)
        x = self.sigmoid(x)
        return x