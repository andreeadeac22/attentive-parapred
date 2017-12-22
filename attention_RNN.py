import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F

class AttentionRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(AttentionRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # for Ux_j - Ux_j is the same fully-connected layer (multiplication by U), time-distirbuted
        self.conv = nn.Conv1d(input_size, hidden_size, 1)

        self.lstm_cell = nn.LSTMCell(hidden_size, hidden_size)

        self.fc1 = nn.Linear(hidden_size, hidden_size)

        self.attn = nn.Linear(hidden_size, input_size) # Wh_t + Ux_j


    def forward(self, input, last_hidden_state):
        W_a = self.fc1(last_hidden_state)
        U_a = self.conv(input)

        attn_weights = F.softmax(
            self.attn(torch.cat(U_a, W_a), 1), dim=1)

        context = torch.bmm(attn_weights,
                                 input)

        output, hidden = self.lstm_cell(last_hidden_state, context)
        return hidden


    def initHidden(self):
        result = Variable(torch.zeros(1, 1, self.hidden_size))
        if use_cuda:
            return result.cuda()
        else:
            return result