from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from preprocessing import NUM_FEATURES

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv1d(28, 3)

        self.bidir_lstm = nn.LSTM(256, bidirectional = True)

        self.dropout = nn.Dropout(0.3)



        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Conv1D, elu, l2 regularizer
        x = F.elu(self.conv1(x))

        #Add residual connections

        #Bidirectional LSTM, dropout, recurrent dropout   -- need to batch - edgar's doing 32.
        # probably need to sort and batch sequences with equal number of residues
        x = self.bidir_lstm(x)

        #Dropout
        x = self.dropout(x)

        #Time-distributed?, dense, sigmoid, l2 regularizer

        return x


input = Variable())

def ab_seq_model(max_cdr_len):
    input_ab = Input(shape=(max_cdr_len, NUM_FEATURES))
    label_mask = Input(shape=(max_cdr_len,))

    seq = MaskingByLambda(mask_by_input(label_mask))(input_ab)
    loc_fts = MaskedConvolution1D(28, 3, padding='same', activation='elu',
                                  kernel_regularizer=l2(0.01))(seq)

    fts = add([seq, loc_fts])

    glb_fts = Bidirectional(LSTM(256, dropout=0.15, recurrent_dropout=0.2,
                                 return_sequences=True),
                            merge_mode='concat')(fts)

    fts = Dropout(0.3)(glb_fts)
    probs = TimeDistributed(Dense(1, activation='sigmoid',
                                     kernel_regularizer=l2(0.01)))(fts)
    model = Model(inputs=[input_ab, label_mask], outputs=probs)

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['binary_accuracy', false_pos, false_neg],
                  sample_weight_mode="temporal")
    return model