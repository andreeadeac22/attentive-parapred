import numpy as np
from sklearn.model_selection import KFold

from torch.autograd import Variable
import torch
import torch.nn as nn
from model import *
import torch.optim as optim
from torch import squeeze
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from torch import index_select

import pickle

from model import AbSeqModel

use_cuda = torch.cuda.is_available()
weights_template="weights-fold-{}.h5"

def simple_run(model, cdrs, lbls, masks, lengths, weights_template_number):
    epochs = 16

    # sample_weight = squeeze((lbls * 1.5 + 1) * masks)
    model = AbSeqModel()

    # create your optimizer
    optimizer1 = optim.Adam(model.parameters(), weight_decay=0.01,
                            lr=0.01)  # l2 regularizer is weight_decay, included in optimizer
    optimizer2 = optim.Adam(model.parameters(), weight_decay=0.01,
                            lr=0.001)  # l2 regularizer is weight_decay, included in optimizer
    criterion = nn.BCELoss()

    total_input = Variable(cdrs)
    total_lbls = lbls

    # example_weight = torch.squeeze((lbls * 1.5 + 1) * masks)

    if use_cuda:
        model.cuda()
        total_input = total_input.cuda()
        total_lbls = total_lbls.cuda()
        masks = masks.cuda()

    loss = 0

    for i in range(epochs):
        if i < 5:
            optimizer = optimizer1
        else:
            optimizer = optimizer2
        for j in range(0, cdrs.shape[0], 32):
            interval = [x for x in range(j, min(cdrs.shape[0], j + 32))]
            interval = torch.LongTensor(interval)
            if use_cuda:
                interval = interval.cuda()
            input = index_select(total_input.data, 0, interval)
            input = Variable(input, requires_grad=True)
            print("j", j)
            print("input shape", input.data.shape)
            # print("lengths", lengths[j:j+32])
            output = model(input, lengths[j:j + 32])
            lbls = index_select(total_lbls, 0, interval)
            print("lbls before pack", lbls.shape)
            lbls = Variable(lbls)

            packed_input = pack_padded_sequence(lbls, lengths[j:j + 32], batch_first=True)

            lbls, _ = pad_packed_sequence(packed_input, batch_first=True)

            print("lbls after pack", lbls.data.shape)
            loss += criterion(output, lbls)
            loss.backward(retain_graph=True)
            optimizer.step()  # Does the update

    torch.save(model.state_dict(), weights_template.format(weights_template_number))
    return model

def kfold_cv_eval(dataset, output_file="crossval-data.p",
                  weights_template="weights-fold-{}.h5", seed=0):
    cdrs, lbls, masks, lengths = dataset["cdrs"], dataset["lbls"], dataset["masks"], dataset["lengths"]
    kf = KFold(n_splits=10, random_state=seed, shuffle=True)

    all_lbls = []
    all_probs = []
    all_masks = []

    for i, (train_idx, test_idx) in enumerate(kf.split(cdrs)):
        print("Fold: ", i + 1)
        print(train_idx)
        print(test_idx)

        lengths_train = [lengths[i] for i in train_idx]
        lengths_test = [lengths[i] for i in test_idx]

        train_idx = torch.from_numpy(train_idx)
        test_idx = torch.from_numpy(test_idx)

        cdrs_train = index_select(cdrs, 0, train_idx)
        lbls_train = index_select(lbls, 0, train_idx)
        mask_train = index_select(masks, 0, train_idx)

        cdrs_test = index_select(cdrs, 0, test_idx)
        lbls_test = index_select(lbls, 0, test_idx)
        mask_test = index_select(masks, 0, test_idx)

        model = AbSeqModel()

        model = simple_run(model, cdrs_train, lbls_train, mask_train, lengths_train, i)

        probs_test = model(cdrs_test, lengths_test)
        all_lbls.append(lbls_test)
        all_probs.append(probs_test)
        all_masks.append(mask_test)

    lbl_mat = torch.concatenate(all_lbls)
    prob_mat = torch.concatenate(all_probs)
    mask_mat = torch.concatenate(all_masks)

    with open(output_file, "wb") as f:
        pickle.dump((lbl_mat, prob_mat, mask_mat), f)