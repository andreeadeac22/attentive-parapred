import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
from torch import index_select

from constants import *
from preprocessing import NUM_FEATURES, AG_NUM_FEATURES

class EpiAttn(nn.Module):
    def __init__(self, hidden_size):
        super(EpiAttn, self).__init__()
        self.hidden_size = hidden_size

        self.conv1 = nn.Conv1d(AG_NUM_FEATURES, AG_NUM_FEATURES, 3, padding=1)
        self.elu = nn.ELU()
        self.dropout1 = nn.Dropout(0.15)
        self.dropout2 = nn.Dropout(0.3)
        self.dropout = nn.Dropout()
        self.conv2 = nn.Conv1d(hidden_size+AG_NUM_FEATURES, 1, 1)
        #self.conv3 = nn.Conv1d(34, 1, 1)

        # for Ux_j - Ux_j is the same fully-connected layer (multiplication by U), time-distirbuted
        self.conv = nn.Conv1d(AG_NUM_FEATURES, 1, 1)

        self.lstm_cell = nn.LSTMCell(AG_NUM_FEATURES, hidden_size)

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
            for names, param in m.named_parameters():
                for name in filter(lambda n: "bias" in n, names):
                    bias = getattr(m, name)
                    n = bias.size(0)
                    start, end = n // 4, n // 2
                    bias.data[start:end].fill_(1.0)
            m.bias_ih.data.fill_(0.0)
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform(m.weight.data)
            m.bias.data.fill_(0.0)

    def forward(self, input, unpacked_masks):
        #print("input shape", input.data.shape);
        bias_mat = 1e9 * (unpacked_masks - 1.0)

        #AbSeqModel
        initial = input
        x = input
        x = torch.transpose(x, 1, 2)
        x = self.conv1(x)
        x = torch.transpose(x, 1, 2)
        x = torch.mul(x, unpacked_masks)
        x = self.elu(x)
        x = x + initial

        #print("x.shape", x.data.shape)
        #print("masks.shape", unpacked_masks.data.shape)
        #print("bias_mat", bias_mat.data)

        # x is batch, time, features

        #Attention
        x = torch.transpose(x, 1, 2)
        u_a = self.conv(x)
        u_a = torch.transpose(u_a, 1, 2)

        x = torch.transpose(x, 1, 2)

        # u_a is batch, time, 1

        all_hidden = []
        hidden = Variable(torch.zeros(x.data.shape[0], self.hidden_size))
        cell = Variable(torch.zeros(x.data.shape[0], self.hidden_size))

        if use_cuda:
            hidden = hidden.cuda()
            cell = cell.cuda()
        #timesteps = 32 # how big the output needs to be - 1 for each residue

        #print("u_a", u_a.shape)

        for i in range(input.data.shape[1]):
            # hidden is batch, features

            hidden = self.dropout(hidden)
            w_a = self.fc1(hidden)
            # w_a is batch, 1
            w_a = w_a.view(w_a.data.shape[0], w_a.data.shape[1], 1)

            #print("w_a", w_a, file=attention_file)

            #print("u_a", u_a.data.shape)
            #print("w_a", w_a.data.shape)

            attn_weights = u_a + w_a # attn_weights is batch, time, 1
            #print("attn.shape after sum", attn_weights.data.shape)

            #print("after sum u+w", attn_weights, file=attention_file)

            attn_weights = F.leaky_relu(attn_weights)
            #print("after leaky", attn_weights, file=attention_file)

            bias_mat = bias_mat.view(attn_weights.data.shape[0], x.data.shape[1], 1)

            attn_weights = attn_weights + bias_mat

            #print("attn_weights before softmax", attn_weights.data, file=attention_file)

            attn_weights = F.softmax(attn_weights, dim=0) # attn_weights is batch, time, 1

            #print("attn_weights after softmax", attn_weights.data.shape)


            attn_weights_transposed = torch.transpose(attn_weights, 1, 2)
            context = torch.bmm(attn_weights_transposed, x) # batch, time, 1 * batch, time, features
            # context is batch, features

            #print("context.shape", context.data.shape)

            context = torch.squeeze(context)

            #print("context.shape", context.data.shape)
            #print("hidden.shape", hidden.data.shape)
            #print("cell.shape", cell.data.shape)

            if use_cuda:
                context = context.cuda()

            hidden, cell = self.lstm_cell(context, (hidden, cell)) #B x in, (B x hid, B x hid)

            all_hidden.append(hidden)

        #AbSeqModel
        all_hidden = torch.stack(all_hidden)
        all_hidden = torch.transpose(all_hidden, 0, 1)

        #print("all_hidden", all_hidden, file=attention_file)
        #print("all_hidden shape", all_hidden.data.shape)

        if use_cuda:
            all_hidden = all_hidden.cuda()

        #print("x shape", x.shape)

        x = self.dropout2(all_hidden)
        x = torch.mul(x, unpacked_masks)

        x = torch.cat((x, input), 2)

        #print("x shape", x.shape)

        #input_transposed = torch.transpose(input, 1, 2)

        x = torch.transpose(x, 1, 2)
        x = self.conv2(x)

        #x +=input_transposed
        #x = self.conv3(x)
        x = torch.transpose(x, 1, 2)
        x = torch.mul(x, unpacked_masks)

        #print("x at the end", x.data, file=attention_file)

        return x