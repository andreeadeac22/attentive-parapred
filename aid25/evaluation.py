from __future__ import print_function
from sklearn.model_selection import KFold

import numpy as np
np.set_printoptions(threshold=np.nan)
from torch.autograd import Variable
import torch
torch.set_printoptions(threshold=50000)
import torch.nn as nn
from model import *
import torch.optim as optim
from torch.optim import lr_scheduler
from torch import squeeze
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from torch import index_select
from sklearn.metrics import confusion_matrix, roc_auc_score, matthews_corrcoef
import pickle

from model import *
from constants import *
from attention_RNN import *
from atrous import *

def sort_batch(cdrs, masks, lengths, lbls):
    order = np.argsort(lengths)
    order = order.tolist()
    order.reverse()
    lengths.sort(reverse=True)
    index = Variable(torch.LongTensor(order))
    if use_cuda:
        index = index.cuda()

    cdrs = torch.index_select(cdrs, 0, index)
    lbls = torch.index_select(lbls, 0, index)
    masks = torch.index_select(masks, 0, index)
    return cdrs, masks, lengths, lbls


def permute_training_data(cdrs, masks, lengths, lbls):
    index = torch.randperm(cdrs.shape[0])
    if use_cuda:
        index = index.cuda()

    cdrs = torch.index_select(cdrs, 0, index)
    lbls = torch.index_select(lbls, 0, index)
    masks = torch.index_select(masks, 0, index)
    lengths = [lengths[i] for i in index]

    return cdrs, masks, lengths, lbls


def simple_run(cdrs_train, lbls_train, masks_train, lengths_train, weights_template, weights_template_number,
               cdrs_test, lbls_test, masks_test, lengths_test):
    print("simple run", file=print_file)
    model = AbSeqModel()

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

    # batch size: increase for speed


    for epoch in range(epochs):
        model.train(True)

        scheduler.step()
        epoch_loss = 0

        batches_done =0

        total_input, total_masks, total_lengths, total_lbls = \
            permute_training_data(total_input, total_masks, total_lengths, total_lbls)

        for j in range(0, cdrs_train.shape[0], 32):
            batches_done+=1
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

            packed_lbls = pack_padded_sequence(lbls, lengths, batch_first=True)
            lbls, _ = pad_packed_sequence(packed_lbls, batch_first=True)

            output = model(input, unpacked_masks, masks, lengths)

            loss_weights = (lbls * 1.5 + 1) * masks
            max_val = (-output).clamp(min=0)
            loss = loss_weights * (output - output * lbls + max_val + ((-max_val).exp() + (-output - max_val).exp()).log())
            masks_added = masks.sum()
            loss = loss.sum() / masks_added

            #print("Epoch %d - Batch %d has loss %d " % (epoch, j, loss.data), file=monitoring_file)
            epoch_loss +=loss

            model.zero_grad()

            loss.backward()
            optimizer.step()
        print("Epoch %d - loss is %f : " % (epoch, epoch_loss.data[0]/batches_done), file=monitoring_file)

        model.eval()

        cdrs_test1, masks_test1, lengths_test1, lbls_test1 = sort_batch(cdrs_test, masks_test, list(lengths_test), lbls_test)

        unpacked_masks_test1 = masks_test1
        packed_input1 = pack_padded_sequence(masks_test1, lengths_test1, batch_first=True)
        masks_test1, _ = pad_packed_sequence(packed_input1, batch_first=True)

        probs_test1 = model(cdrs_test1, unpacked_masks_test1, masks_test1, lengths_test1)

        # K.mean(K.equal(lbls_test, K.round(y_pred)), axis=-1)

        sigmoid = nn.Sigmoid()
        probs_test2 = sigmoid(probs_test1)
        probs_test3 = sigmoid(probs_test1)*masks_test1

        """""
        for i in range(probs_test3.data.shape[0]):
            if(probs_test2[i] != probs_test3):
                print("They are different")
                print("masks_test", masks_test1)
                print("lengths", lengths_test1)
         """


        # multiplying with masks helps???
        # problem with masks, lengths or flatten?


        # Mask is 0 for chains with 1 residue - TODO 

        probs_test2 = probs_test2.data.cpu().numpy().astype('float32')
        probs_test3 = probs_test3.data.cpu().numpy().astype('float32')
        lbls_test1 = lbls_test1.data.cpu().numpy().astype('int32')

        probs_test2 = flatten_with_lengths(probs_test2, lengths_test1)
        probs_test3 = flatten_with_lengths(probs_test3, lengths_test1)

        lbls_test1 = flatten_with_lengths(lbls_test1, lengths_test1)


        print("Roc", roc_auc_score(lbls_test1, probs_test2))
        print("Roc with masks", roc_auc_score(lbls_test1, probs_test3))





    torch.save(model.state_dict(), weights_template.format(weights_template_number))

    print("test", file=track_f)

    model.eval()

    cdrs_test, masks_test, lengths_test, lbls_test = sort_batch(cdrs_test, masks_test, list(lengths_test), lbls_test)

    unpacked_masks_test = masks_test
    packed_input = pack_padded_sequence(masks_test, lengths_test, batch_first=True)
    masks_test, _ = pad_packed_sequence(packed_input, batch_first=True)

    probs_test = model(cdrs_test, unpacked_masks_test, masks_test, lengths_test)

    #K.mean(K.equal(lbls_test, K.round(y_pred)), axis=-1)

    sigmoid = nn.Sigmoid()
    probs_test = sigmoid(probs_test)

    return probs_test, lbls_test

def attention_run(cdrs_train, lbls_train, masks_train, lengths_train, weights_template, weights_template_number,
               cdrs_test, lbls_test, masks_test, lengths_test):
    print("attention run", file=print_file)
    model = AttentionRNN(32, 512)

    model.train()

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
        print("Epoch %d - loss is %f : " % (epoch, epoch_loss.data[0]), file=monitoring_file)

    torch.save(model.state_dict(), weights_template.format(weights_template_number))

    print("test", file=track_f)

    model.eval()

    cdrs_test, masks_test, lengths_test, lbls_test = sort_batch(cdrs_test, masks_test, lengths_test, lbls_test)

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

    #K.mean(K.equal(lbls_test, K.round(y_pred)), axis=-1)

    sigmoid = nn.Sigmoid()
    probs_test = sigmoid(probs_test)

    return probs_test, lbls_test

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

def kfold_cv_eval(dataset, output_file="crossval-data.p",
                  weights_template="weights-fold-{}.h5", seed=0):
    cdrs, lbls, masks, lengths = dataset["cdrs"], dataset["lbls"], dataset["masks"], dataset["lengths"]

    #print("cdrs", cdrs, file=data_file)
    #print("lbls", lbls, file=data_file)
    #print("masks", masks, file=data_file)
    #print("lengths", lengths, file=data_file)

    kf = KFold(n_splits=NUM_SPLIT, random_state=seed, shuffle=True)

    all_lbls = []
    all_probs = []
    all_masks = []

    for i, (train_idx, test_idx) in enumerate(kf.split(cdrs)):
        print("Fold: ", i + 1)
        #print(train_idx, )
        #print(test_idx)

        lengths_train = [lengths[i] for i in train_idx]
        lengths_test = [lengths[i] for i in test_idx]

        train_idx = torch.from_numpy(train_idx)
        test_idx = torch.from_numpy(test_idx)

        cdrs_train = index_select(cdrs, 0, train_idx)
        lbls_train = index_select(lbls, 0, train_idx)
        mask_train = index_select(masks, 0, train_idx)

        cdrs_test = Variable(index_select(cdrs, 0, test_idx))
        lbls_test = Variable(index_select(lbls, 0, test_idx))
        mask_test = Variable(index_select(masks, 0, test_idx))

        probs_test, lbls_test = simple_run(cdrs_train, lbls_train, mask_train, lengths_train, weights_template, i,
                                cdrs_test, lbls_test, mask_test, lengths_test)

        #probs_test, lbls_test = attention_run(cdrs_train, lbls_train, mask_train, lengths_train, weights_template, i,
        #                      cdrs_test, lbls_test, mask_test, lengths_test)


        #probs_test, lbls_test = atrous_run(cdrs_train, lbls_train, mask_train, lengths_train, weights_template, i,
        #                         cdrs_test, lbls_test, mask_test, lengths_test)

        print("test", file=track_f)

        print("probs_test", probs_test, file=track_f)

        all_lbls.append(lbls_test)

        probs_test_pad = torch.zeros(probs_test.data.shape[0], MAX_CDR_LENGTH, probs_test.data.shape[2])
        probs_test_pad[:probs_test.data.shape[0], :probs_test.data.shape[1], :] = probs_test.data
        probs_test_pad = Variable(probs_test_pad)

        all_probs.append(probs_test_pad)
        all_masks.append(mask_test)

    lbl_mat = torch.cat(all_lbls)
    prob_mat = torch.cat(all_probs)
    mask_mat = torch.cat(all_masks)

    with open(output_file, "wb") as f:
        pickle.dump((lbl_mat, prob_mat, mask_mat), f)

def flatten_with_lengths(matrix, lengths):
    seqs = []
    for i, example in enumerate(matrix):
        seq = example[:lengths[i]]
        seqs.append(seq)
    return np.concatenate(seqs)

def compute_classifier_metrics(labels, probs, threshold=0.5):
    matrices = []
    aucs = []
    mcorrs = []

    for l, p in zip(labels, probs):
        aucs.append(roc_auc_score(l, p))
        l_pred = (p > threshold).astype(int)
        matrices.append(confusion_matrix(l, l_pred))
        mcorrs.append(matthews_corrcoef(l, l_pred))

    matrices = np.stack(matrices)
    mean_conf = np.mean(matrices, axis=0)
    errs_conf = 2 * np.std(matrices, axis=0)

    tps = matrices[:, 1, 1]
    fns = matrices[:, 1, 0]
    fps = matrices[:, 0, 1]

    tpsf = tps.astype(float)
    fnsf= fns.astype(float)
    fpsf = fps.astype(float)

    recalls = tpsf / (tpsf + fnsf)
    precisions = tpsf / (tpsf + fpsf)

    rec = np.mean(recalls)
    rec_err = 2 * np.std(recalls)
    prec = np.mean(precisions)
    prec_err = 2 * np.std(precisions)

    fscores = 2 * precisions * recalls / (precisions + recalls)
    fsc = np.mean(fscores)
    fsc_err = 2 * np.std(fscores)

    auc_scores = np.array(aucs)
    auc = np.mean(auc_scores)
    auc_err = 2 * np.std(auc_scores)

    mcorr_scores = np.array(mcorrs)
    mcorr = np.mean(mcorr_scores)
    mcorr_err = 2 * np.std(mcorr_scores)

    print("Mean confusion matrix and error")
    print(mean_conf)
    print(errs_conf)

    print("Recall = {} +/- {}".format(rec, rec_err))
    print("Precision = {} +/- {}".format(prec, prec_err))
    print("F-score = {} +/- {}".format(fsc, fsc_err))
    print("ROC AUC = {} +/- {}".format(auc, auc_err))
    print("MCC = {} +/- {}".format(mcorr, mcorr_err))

def open_crossval_results(folder="cv-ab-seq", num_results=NUM_ITERATIONS,
                          loop_filter=None, flatten_by_lengths=True):
    class_probabilities = []
    labels = []

    for r in range(num_results):
        result_filename = "{}/run-{}.p".format(folder, r)
        with open(result_filename, "rb") as f:
            lbl_mat, prob_mat, mask_mat = pickle.load(f)

            lbl_mat = lbl_mat.data.cpu().numpy()
            prob_mat = prob_mat.data.cpu().numpy()
            mask_mat = mask_mat.data.cpu().numpy()

        # Get entries corresponding to the given loop
        if loop_filter is not None:
            lbl_mat = lbl_mat[loop_filter::6]
            prob_mat = prob_mat[loop_filter::6]
            mask_mat = mask_mat[loop_filter::6]

        if not flatten_by_lengths:
            class_probabilities.append(prob_mat)
            labels.append(lbl_mat)
            continue

        # Discard sequence padding
        seq_lens = np.sum(np.squeeze(mask_mat), axis=1)
        seq_lens = seq_lens.astype(int)

        p = flatten_with_lengths(prob_mat, seq_lens)
        l = flatten_with_lengths(lbl_mat, seq_lens)

        class_probabilities.append(p)
        labels.append(l)

    return labels, class_probabilities

