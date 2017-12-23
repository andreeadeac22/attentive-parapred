import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F

from constants import *

class AttentionRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(AttentionRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # for Ux_j - Ux_j is the same fully-connected layer (multiplication by U), time-distirbuted
        self.conv = nn.Conv1d(input_size, 1, 1)

        self.lstm_cell = nn.LSTMCell(hidden_size, hidden_size)

        self.fc1 = nn.Linear(hidden_size, 1)


    def forward(self, input, bias_mat):
        u_a = self.conv(input)
        all_hidden = []
        hidden = Variable(torch.zeros(1,1))  #TODO: decide dimesnions for hidden
        context = Variable(torch.zeros(1,1))


        if use_cuda:
            hidden = hidden.cuda()

        for i in range(MAX_CDR_LENGTH):
            w_a = self.fc1(hidden)
            attn_weights = F.softmax(nn.LeakyRelu(u_a+w_a)+bias_mat)

            context = torch.bmm(attn_weights,
                                     input)

            hidden, context = self.lstm_cell(context, (hidden, context))
            all_hidden.append(hidden)
        return all_hidden