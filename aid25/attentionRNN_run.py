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
from sklearn.metrics import confusion_matrix, roc_auc_score, matthews_corrcoef

from attention_RNN import *
from constants import *
from evaluation_tools import *

def attention_run(cdrs_train, lbls_train, masks_train, lengths_train, weights_template, weights_template_number,
               cdrs_test, lbls_test, masks_test, lengths_test):
    print("attention run", file=print_file)
    model = AttentionRNN(512)

    ignored_params = list(map(id, [model.conv1.weight, model.conv2.weight]))
    base_params = filter(lambda p: id(p) not in ignored_params,
                         model.parameters())

    optimizer = optim.Adam([
        {'params': base_params},
        {'params': model.conv1.weight, 'weight_decay': 0.01},
        {'params': model.conv2.weight, 'weight_decay': 0.01}
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
        model.train(True)

        scheduler.step()
        epoch_loss = 0

        batches_done =0

        total_input, total_masks, total_lengths, total_lbls = \
            permute_training_data(total_input, total_masks, total_lengths, total_lbls)

        for j in range(0, cdrs_train.shape[0], batch_size):
            batches_done+=1
            interval = [x for x in range(j, min(cdrs_train.shape[0], j + batch_size))]
            interval = torch.LongTensor(interval)
            if use_cuda:
                interval = interval.cuda()

            input = Variable(index_select(total_input, 0, interval), requires_grad=True)
            masks = Variable(index_select(total_masks, 0, interval))
            lengths = total_lengths[j:j + batch_size]
            lbls = Variable(index_select(total_lbls, 0, interval))

            input, masks, lengths, lbls = sort_batch(input, masks, list(lengths), lbls)

            unpacked_masks = masks

            packed_masks = pack_padded_sequence(masks, lengths, batch_first=True)
            masks, _ = pad_packed_sequence(packed_masks, batch_first=True)

            unpacked_lbls = lbls

            packed_lbls = pack_padded_sequence(lbls, lengths, batch_first=True)
            lbls, _ = pad_packed_sequence(packed_lbls, batch_first=True)

            bias_mat = torch.FloatTensor(input.data.shape[0], MAX_CDR_LENGTH)
            bias_mat.fill_(-1e9)
            #print("unpacked_masks.data.shape[0]", unpacked_masks.data.shape[0])
            #print("unpacked_masks.data.shape[1]", unpacked_masks.data.shape[1])
            for um_i in range(0, unpacked_masks.data.shape[0]):
                for um_j in range(0, unpacked_masks.data.shape[1]):
                    #print("unpacked_masks.data[um_i][um_j]", unpacked_masks.data[um_i][um_j])
                    if( unpacked_masks.data[um_i][um_j].cpu().numpy() == 1):
                        bias_mat[um_i, um_j] = 0
            #bias_mat[:unpacked_masks.data.shape[0], :unpacked_masks.data.shape[1], :] = 0
            bias_mat = Variable(bias_mat)
            if use_cuda:
                bias_mat = bias_mat.cuda()

            output = model(input, unpacked_masks, bias_mat)

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
        print("Epoch %d - loss is %f : " % (epoch, epoch_loss.data[0]/batches_done))

        model.eval()

        cdrs_test1, masks_test1, lengths_test1, lbls_test1 = sort_batch(cdrs_test, masks_test, list(lengths_test),
                                                                    lbls_test)

        unpacked_masks_test1 = masks_test1
        packed_input1 = pack_padded_sequence(masks_test1, lengths_test1, batch_first=True)
        masks_test1, _ = pad_packed_sequence(packed_input1, batch_first=True)

        bias_mat1 = torch.FloatTensor(cdrs_test.data.shape[0], MAX_CDR_LENGTH)
        bias_mat1.fill_(-1e9)
        # print("unpacked_masks.data.shape[0]", unpacked_masks.data.shape[0])
        # print("unpacked_masks.data.shape[1]", unpacked_masks.data.shape[1])
        for um_i in range(0, unpacked_masks_test1.data.shape[0]):
            for um_j in range(0, unpacked_masks_test1.data.shape[1]):
                # print("unpacked_masks.data[um_i][um_j]", unpacked_masks.data[um_i][um_j])
                if (unpacked_masks_test1.data[um_i][um_j].cpu().numpy() == 1):
                    bias_mat1[um_i, um_j] = 0
        # bias_mat[:unpacked_masks.data.shape[0], :unpacked_masks.data.shape[1], :] = 0
        bias_mat1 = Variable(bias_mat1)
        if use_cuda:
            bias_mat1 = bias_mat1.cuda()

        probs_test1 = model(cdrs_test1, unpacked_masks_test1, bias_mat1)

        sigmoid = nn.Sigmoid()
        probs_test2 = sigmoid(probs_test1)

        probs_test2 = probs_test2.data.cpu().numpy().astype('float32')
        lbls_test1 = lbls_test1.data.cpu().numpy().astype('int32')

        probs_test2 = flatten_with_lengths(probs_test2, lengths_test1)

        lbls_test1 = flatten_with_lengths(lbls_test1, lengths_test1)

        print("Roc", roc_auc_score(lbls_test1, probs_test2))
        print("Accuracy", np.mean(np.equal(lbls_test1, np.round(probs_test2))))

    torch.save(model.state_dict(), weights_template.format(weights_template_number))

    print("test", file=track_f)

    model.eval()

    cdrs_test, masks_test, lengths_test, lbls_test = sort_batch(cdrs_test, masks_test, list(lengths_test), lbls_test)

    unpacked_masks_test = masks_test
    packed_input = pack_padded_sequence(masks_test, lengths_test, batch_first=True)
    masks_test, _ = pad_packed_sequence(packed_input, batch_first=True)

    bias_mat = torch.FloatTensor(cdrs_test.data.shape[0], MAX_CDR_LENGTH)
    bias_mat.fill_(-1e9)
    # print("unpacked_masks.data.shape[0]", unpacked_masks.data.shape[0])
    # print("unpacked_masks.data.shape[1]", unpacked_masks.data.shape[1])
    for um_i in range(0, unpacked_masks_test.data.shape[0]):
        for um_j in range(0, unpacked_masks_test.data.shape[1]):
            # print("unpacked_masks.data[um_i][um_j]", unpacked_masks.data[um_i][um_j])
            if (unpacked_masks_test.data[um_i][um_j].cpu().numpy() == 1):
                bias_mat[um_i, um_j] = 0
    # bias_mat[:unpacked_masks.data.shape[0], :unpacked_masks.data.shape[1], :] = 0
    bias_mat = Variable(bias_mat)
    if use_cuda:
        bias_mat = bias_mat.cuda()

    probs_test = model(cdrs_test, unpacked_masks_test, bias_mat)

    sigmoid = nn.Sigmoid()
    probs_test = sigmoid(probs_test)

    probs_test4 = sigmoid(probs_test)
    probs_test5 = sigmoid(probs_test) * unpacked_masks_test1

    probs_test4 = probs_test4.data.cpu().numpy().astype('float32')
    probs_test5 = probs_test5.data.cpu().numpy().astype('float32')
    lbls_test3 = lbls_test.data.cpu().numpy().astype('int32')

    probs_test4 = flatten_with_lengths(probs_test4, lengths_test)
    probs_test5 = flatten_with_lengths(probs_test5, lengths_test)

    lbls_test3 = flatten_with_lengths(lbls_test3, lengths_test)

    print("Roc", roc_auc_score(lbls_test3, probs_test4))
    print("Roc with masks", roc_auc_score(lbls_test3, probs_test5))
    print("Accuracy", np.mean(np.equal(lbls_test3, np.round(probs_test4))))

    return probs_test, lbls_test, probs_test4, lbls_test3