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
from torch import squeeze
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from torch import index_select
from sklearn.metrics import confusion_matrix, roc_auc_score, matthews_corrcoef
import pickle

from model import *
from constants import *

def simple_run(cdrs_train, lbls_train, masks_train, lengths_train, weights_template, weights_template_number,
               cdrs_test, lbls_test, masks_test, lengths_test):
    #print("simple run - weights_template", weights_template)
    print("simple run", file=print_file)
    model = AbSeqModel()

    # create your optimizer
    optimizer1 = optim.Adam(model.parameters(), weight_decay=0.01,
                            lr=0.01)  # l2 regularizer is weight_decay, included in optimizer
    optimizer2 = optim.Adam(model.parameters(), weight_decay=0.01,
                            lr=0.001)  # l2 regularizer is weight_decay, included in optimizer
    criterion = nn.BCELoss()

    total_input = Variable(cdrs_train)
    total_lbls = lbls_train
    total_masks = masks_train

    if use_cuda:
        model.cuda()
        total_input = total_input.cuda()
        total_lbls = total_lbls.cuda()
        total_masks = total_masks.cuda()
        cdrs_test = cdrs_test.cuda()
        lbls_test = lbls_test.cuda()
        masks_test = masks_test.cuda()

    loss = 0

    for i in range(epochs):
        if i < 5:
            optimizer = optimizer1
        else:
            optimizer = optimizer2
        epoch_loss = 0
        print("Epoch: ", i, file=monitoring_file)
        for j in range(0, cdrs_train.shape[0], 32):
            interval = [x for x in range(j, min(cdrs_train.shape[0], j + 32))]
            interval = torch.LongTensor(interval)
            if use_cuda:
                interval = interval.cuda()
            input = index_select(total_input.data, 0, interval)
            input = Variable(input, requires_grad=True)
            masks = Variable(index_select(total_masks, 0, interval))
            print("train - j", j, file=print_file)

            #print("input shape", input.data.shape)
            # print("lengths", lengths[j:j+32])

            unpacked_masks = masks

            packed_input = pack_padded_sequence(masks, lengths_train[j:j + 32], batch_first=True)

            masks, _ = pad_packed_sequence(packed_input, batch_first=True)

            output = model(input, unpacked_masks, masks, lengths_train[j:j + 32])
            lbls = index_select(total_lbls, 0, interval)
            #print("lbls before pack", lbls.shape)
            lbls = Variable(lbls)

            packed_input = pack_padded_sequence(lbls, lengths_train[j:j + 32], batch_first=True)

            lbls, _ = pad_packed_sequence(packed_input, batch_first=True)

            #print("lbls after pack", lbls.data.shape)
            #loss = criterion(output, lbls)

            #print("sigmoid(output)", F.sigmoid(output), file=print_file)
            print("lbls", lbls, file=print_file)
            print("output", output, file=print_file)

            loss_weights = (lbls * 1.5 + 1) * masks
            max_val = (-output).clamp(min=0)
            loss = loss_weights * (output - output * lbls + max_val + ((-max_val).exp() + (-output - max_val).exp()).log())
            #loss = - (lbls*torch.log(output) + (1-lbls)*torch.log(1-output))
            print("loss before sum", loss, file=print_file)

            print("loss size", loss.data.shape, file=print_file)

            print("masks", masks, file=print_file)
            print("masks size", masks.data.shape, file=print_file)

            #loss = loss * masks

            print("loss after multi", loss, file=print_file)

            masks_added = masks.sum()

            loss = loss.sum() / masks_added
            #loss = loss / masks_added


            print("Batch: ", j, file=monitoring_file)
            print("Loss: ", loss.data, file=monitoring_file)

            epoch_loss +=loss
            loss.backward()
            optimizer.step()  # Does the update
        print("Loss at the end of epoch: ", epoch_loss, file=monitoring_file)

    print("model.state_dict().keys()", model.state_dict().keys)
    torch.save(model.state_dict(), weights_template.format(weights_template_number))

    print("test", file=track_f)
    unpacked_masks_test = masks_test
    packed_input = pack_padded_sequence(masks_test, lengths_test, batch_first=True)
    masks_test, _ = pad_packed_sequence(packed_input, batch_first=True)
    probs_test = model(cdrs_test, unpacked_masks_test, masks_test, lengths_test)

    return probs_test

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
        #print(train_idx)
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

        probs_test = simple_run(cdrs_train, lbls_train, mask_train, lengths_train, weights_template, i,
                                cdrs_test, lbls_test, mask_test, lengths_test)
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

            print("new lbl_mat", lbl_mat, file=print_file)
            print("new prob_mat", prob_mat, file=print_file)
            print("mask_mat", mask_mat, file=print_file)

            print("new lbl_mat.shape", lbl_mat.shape, file=print_file)
            print("new prob_mat.shape", prob_mat.shape, file=print_file)
            print("new mask_mat.shape", mask_mat.shape, file=print_file)

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
        print("seq_lens", seq_lens, file=print_file)
        seq_lens = seq_lens.astype(int)

        p = flatten_with_lengths(prob_mat, seq_lens)
        l = flatten_with_lengths(lbl_mat, seq_lens)

        class_probabilities.append(p)
        labels.append(l)

    return labels, class_probabilities

