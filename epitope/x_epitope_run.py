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

from epitope_model import *
from constants import *
from evaluation_tools import *
from epitope_cross import *

def x_epitope_run(epi_train, lbls_train, masks_train, lengths_train, cdrs_train, cdr_masks_train, cdr_lengths_train,
                  weights_template, weights_template_number, epi_test, lbls_test, masks_test, lengths_test,
                  cdrs_test, cdr_masks_test, cdr_lengths_test):

    print("epitope run", file=print_file)
    model = EpitopeX()

    ignored_params = list(map(id, [model.conv1.weight,
                                   #model.conv2.weight, model.conv3.weight, model.aconv1.weight,
                                   #model.aconv2.weight
                                ]))
    base_params = filter(lambda p: id(p) not in ignored_params,
                         model.parameters())

    optimizer = optim.Adam([
        {'params': base_params},
        {'params': model.conv1.weight, 'weight_decay': 0.01},
        #{'params': model.conv2.weight, 'weight_decay': 0.01},
        #{'params': model.conv3.weight, 'weight_decay': 0.01},
        #{'params': model.aconv1.weight, 'weight_decay': 0.01},
        #{'params': model.aconv2.weight, 'weight_decay': 0.01}
    ], lr=0.01)

    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10], gamma=0.1)

    total_input = epi_train
    total_lbls = lbls_train
    total_masks = masks_train
    total_lengths = lengths_train

    total_cdrs_train = cdrs_train
    total_cdr_masks_train = cdr_masks_train
    total_cdr_lengths_train = cdr_lengths_train

    if use_cuda:
        print("using cuda")
        model.cuda()
        total_input = total_input.cuda()
        total_lbls = total_lbls.cuda()
        total_masks = total_masks.cuda()
        epi_test = epi_test.cuda()
        lbls_test = lbls_test.cuda()
        masks_test = masks_test.cuda()

        total_cdrs_train = total_cdrs_train.cuda()
        total_cdr_masks_train = total_cdr_masks_train.cuda()
        cdrs_test = cdrs_test.cuda()
        cdr_masks_test = cdr_masks_test.cuda()

    for epoch in range(epochs):
        model.train(True)
        scheduler.step()
        epoch_loss = 0

        batches_done=0

        #total_input, total_masks, total_lengths, total_lbls = \
        #    permute_training_data(total_input, total_masks, total_lengths, total_lbls)

        total_cdrs_train, total_cdr_masks_train, total_cdr_lengths_train, total_input, total_masks, total_lengths, total_masks = \
            permute_training_cross_data(cdrs=total_cdrs_train, cdr_masks=total_cdr_masks_train, cdr_lengths=total_cdr_lengths_train,
                        ag=total_input, ag_masks=total_masks, ag_lengths=total_lengths, ag_lbls=total_lbls)

        for j in range(0, epi_train.shape[0], batch_size):
            batches_done +=1
            interval = [x for x in range(j, min(epi_train.shape[0], j + batch_size))]
            interval = torch.LongTensor(interval)
            if use_cuda:
                interval = interval.cuda()

            input = Variable(index_select(total_input, 0, interval), requires_grad=True)
            masks = Variable(index_select(total_masks, 0, interval))
            lengths = total_lengths[j:j + batch_size]
            lbls = Variable(index_select(total_lbls, 0, interval))

            cdrs_train = Variable(index_select(total_cdrs_train, 0, interval))
            cdr_masks_train = Variable(index_select(total_cdr_masks_train, 0, interval))
            cdr_lengths_train = total_cdr_lengths_train[j:j+batch_size]

            cdrs, cdr_masks, input, masks, lbls, lengths = sort_cross_batch(cdrs=cdrs_train, cdr_masks=cdr_masks_train,
                                               ag=input, ag_masks=masks, ag_lengths=list(lengths), ag_lbls=lbls)

            unpacked_masks = masks

            packed_masks = pack_padded_sequence(masks, lengths, batch_first=True)
            masks, _ = pad_packed_sequence(packed_masks, batch_first=True)

            unpacked_lbls = lbls

            packed_lbls = pack_padded_sequence(lbls, lengths, batch_first=True)
            lbls, _ = pad_packed_sequence(packed_lbls, batch_first=True)


            output, attn_coeff = model(input, unpacked_masks, cdrs, cdr_masks)

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

        cdrs_test2, cdr_masks_test2, epi_test2, masks_test2, lbls_test2, lengths_test2 = \
            sort_cross_batch(cdrs=cdrs_test, cdr_masks=cdr_masks_test, ag=epi_test, ag_masks=masks_test,
                                        ag_lengths=list(lengths_test), ag_lbls=lbls_test)

        #epi_test2, masks_test2, lengths_test2, lbls_test2 = sort_batch(epi_test, masks_test, list(lengths_test),
        #                                                            lbls_test)

        unpacked_masks_test2 = masks_test2

        probs_test2, attn_coeff_test2 = model(epi_test2, unpacked_masks_test2, cdrs_test2, cdr_masks_test2)

        # K.mean(K.equal(lbls_test, K.round(y_pred)), axis=-1)

        sigmoid = nn.Sigmoid()
        probs_test2 = sigmoid(probs_test2)

        probs_test2 = probs_test2.data.cpu().numpy().astype('float32')
        lbls_test2 = lbls_test2.data.cpu().numpy().astype('int32')

        probs_test2 = flatten_with_lengths(probs_test2, lengths_test2)
        lbls_test2 = flatten_with_lengths(lbls_test2, lengths_test2)

        print("Roc", roc_auc_score(lbls_test2, probs_test2))

    torch.save(model.state_dict(), weights_template.format(weights_template_number))

    print("test", file=track_f)
    model.eval()

    cdrs_test, cdr_masks_test, epi_test, masks_test, lbls_test, lengths_test = \
        sort_cross_batch(cdrs=cdrs_test, cdr_masks=cdr_masks_test, ag=epi_test, ag_masks=masks_test,
                         ag_lengths=list(lengths_test), ag_lbls=lbls_test)

    #epi_test, masks_test, lengths_test, lbls_test = sort_batch(epi_test, masks_test, list(lengths_test), lbls_test)

    unpacked_masks_test = masks_test
    packed_input = pack_padded_sequence(masks_test, list(lengths_test), batch_first=True)
    masks_test, _ = pad_packed_sequence(packed_input, batch_first=True)

    probs_test, attn_coeff_test = model(epi_test, unpacked_masks_test, cdrs_test, cdr_masks_test)

    # K.mean(K.equal(lbls_test, K.round(y_pred)), axis=-1)

    sigmoid = nn.Sigmoid()
    probs_test = sigmoid(probs_test)

    probs_test1 = probs_test.data.cpu().numpy().astype('float32')
    lbls_test1 = lbls_test.data.cpu().numpy().astype('int32')

    probs_test1 = flatten_with_lengths(probs_test1, list(lengths_test))
    lbls_test1 = flatten_with_lengths(lbls_test1, list(lengths_test))

    print("Roc", roc_auc_score(lbls_test1, probs_test1))

    return probs_test, lbls_test, probs_test1, lbls_test1  # get them in kfold, append, concatenate do roc on them