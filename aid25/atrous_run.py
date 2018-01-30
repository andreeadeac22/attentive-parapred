from __future__ import print_function

import numpy as np
np.set_printoptions(threshold=np.nan)
from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from torch import index_select

from atrous import *
from constants import *
from evaluation_tools import *

def atrous_run(cdrs_train, lbls_train, masks_train, lengths_train, weights_template, weights_template_number,
               cdrs_test, lbls_test, masks_test, lengths_test):

    print("dilated run", file=print_file)
    model = DilatedConv()
    model.train()

    ignored_params = list(map(id, [model.conv1.weight]))
    base_params = filter(lambda p: id(p) not in ignored_params,
                         model.parameters())

    optimizer = optim.Adam([
        {'params': base_params},
        {'params': model.conv1.weight, 'weight_decay': 0.01},
    ], lr=0.01)

    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5], gamma=0.1)

    total_input = cdrs_train
    total_lbls = lbls_train
    total_masks = masks_train
    total_lengths = lengths_train

    if use_cuda:
        print("using cuda")
        model.cuda()
        total_input = total_input.cuda()
        total_lbls = total_lbls.cuda()
        total_masks = total_masks.cuda()
        cdrs_test = cdrs_test.cuda()
        lbls_test = lbls_test.cuda()
        masks_test = masks_test.cuda()

    for epoch in range(epochs):
        scheduler.step()
        epoch_loss = 0

        total_input, total_masks, total_lengths, total_lbls = \
            permute_training_data(total_input, total_masks, total_lengths, total_lbls)

        for j in range(0, cdrs_train.shape[0], 32):
            interval = [x for x in range(j, min(cdrs_train.shape[0], j + 32))]
            interval = torch.LongTensor(interval)
            if use_cuda:
                interval = interval.cuda()

            input = Variable(index_select(total_input, 0, interval), requires_grad=True)
            masks = Variable(index_select(total_masks, 0, interval))
            lengths = total_lengths[j:j + 32]
            lbls = Variable(index_select(total_lbls, 0, interval))

            input, masks, lengths, lbls = sort_batch(input, masks, lengths, lbls)

            unpacked_masks = masks

            packed_masks = pack_padded_sequence(masks, lengths, batch_first=True)
            masks, _ = pad_packed_sequence(packed_masks, batch_first=True)

            unpacked_lbls = lbls

            packed_lbls = pack_padded_sequence(lbls, lengths, batch_first=True)
            lbls, _ = pad_packed_sequence(packed_lbls, batch_first=True)


            output = model(input, unpacked_masks)

            loss_weights = (unpacked_lbls * 1.5 + 1) * unpacked_masks
            max_val = (-output).clamp(min=0)
            loss = loss_weights * (output - output * unpacked_lbls + max_val + ((-max_val).exp() + (-output - max_val).exp()).log())
            masks_added = masks.sum()
            loss = loss.sum() / masks_added

            #print("Epoch %d - Batch %d has loss %d " % (epoch, j, loss.data), file=monitoring_file)
            epoch_loss +=loss

            model.zero_grad()

            loss.backward()
            optimizer.step()
        print("Epoch %d - loss is %f : " % (epoch, epoch_loss.data[0]), file=monitoring_file)

    torch.save(model.state_dict(), weights_template.format(weights_template_number))

    print("test", file=track_f)
    model.eval()

    cdrs_test, masks_test, lengths_test, lbls_test = sort_batch(cdrs_test, masks_test, lengths_test, lbls_test)

    unpacked_masks_test = masks_test

    probs_test = model(cdrs_test, unpacked_masks_test)

    #K.mean(K.equal(lbls_test, K.round(y_pred)), axis=-1)

    sigmoid = nn.Sigmoid()
    probs_test = sigmoid(probs_test)

    return probs_test, lbls_test