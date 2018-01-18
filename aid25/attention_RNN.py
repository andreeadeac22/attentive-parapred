import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
from torch import index_select

from constants import *

class AttentionRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(AttentionRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.conv1 = nn.Conv1d(28, 28, 3, padding=1)
        self.elu = nn.ELU()
        self.dropout1 = nn.Dropout(0.15)
        self.dropout2 = nn.Dropout(0.3)
        self.conv2 = nn.Conv1d(512, 1, 1)

        # for Ux_j - Ux_j is the same fully-connected layer (multiplication by U), time-distirbuted
        self.conv = nn.Conv1d(input_size, 1, 1)

        self.lstm_cell = nn.LSTMCell(input_size, hidden_size)

        self.fc1 = nn.Linear(hidden_size, 1)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Conv1d):
            torch.nn.init.xavier_uniform(m.weight.data)
            m.bias.data.fill_(0.0)
        if isinstance(m, nn.LSTMCell):
            torch.nn.init.xavier_uniform(m.weight_ih)
            torch.nn.init.orthogonal(m.weight_hh)
            """""
            for names in m._all_weights:
                for name in filter(lambda n: "bias" in n, names):
                    bias = getattr(m, name)
                    n = bias.size(0)
                    start, end = n // 4, n // 2
                    bias.data[start:end].fill_(1.0)
            """
            m.bias_ih.data.fill_(0.0)


    def forward(self, input, unpacked_masks, bias_mat):

        #AbSeqModel
        initial = input
        x = input
        x_transposed = torch.transpose(x, 1, 2)
        x = torch.transpose(x, 1, 2)
        x = self.conv1(x)
        x = torch.transpose(x, 1, 2)
        x = torch.mul(x, unpacked_masks)
        x = self.elu(x)
        x = x + initial

        #print("x.shape", x.data.shape)
        #print("masks.shape", unpacked_masks.data.shape)
        #print("bias_mat", bias_mat.data)

        #Attention
        u_a = self.conv(x)

        all_hidden = []
        hidden = Variable(torch.zeros(x.data.shape[1], self.hidden_size))
        c_0 = Variable(torch.zeros(x.data.shape[1], self.hidden_size))

        if use_cuda:
            hidden = hidden.cuda()
            c_0 = c_0.cuda()

        for i in range(x.data.shape[0]):
            w_a = self.fc1(hidden)

            attn_weights = F.leaky_relu(u_a+w_a)
            attn_weights = attn_weights + bias_mat
            attn_weights = F.softmax(attn_weights) # decompose this operation and check sum works

            context = torch.bmm(attn_weights, x_transposed)
            if use_cuda:
                context = context.cuda()

            hidden, context = self.lstm_cell(context[i], (hidden, c_0)) #B x in, (B x hid, B x hid)

            c_0 = context
            all_hidden.append(hidden)

        #AbSeqModel
        all_hidden = torch.stack(all_hidden)
        if use_cuda:
            all_hidden = all_hidden.cuda()
        x = self.dropout2(all_hidden)
        x = torch.transpose(x, 1, 2)
        x = self.conv2(x)
        x = torch.transpose(x, 1, 2)
        x = torch.mul(x, unpacked_masks)
        return x