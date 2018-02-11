from __future__ import print_function
from sklearn.model_selection import KFold

import numpy as np
np.set_printoptions(threshold=np.nan)
from torch.autograd import Variable
import torch
torch.set_printoptions(threshold=50000)
from torch import squeeze
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from torch import index_select
from sklearn.metrics import confusion_matrix, roc_auc_score, matthews_corrcoef
import pickle

from epitope_run import *

def kfold_cv_eval(dataset, output_file="crossval-data.p",
                  weights_template="weights-fold-{}.h5", seed=0):
    ag, lbls, masks, lengths = dataset["ag"], dataset["ag_lbls"], dataset["ag_masks"], dataset["ag_lengths"]

    print("ag", ag.shape)
    print("lbls", lbls.shape)
    print("masks", masks.shape)

    kf = KFold(n_splits=NUM_SPLIT, random_state=seed, shuffle=True)

    all_lbls2 = []
    all_probs2 = []
    all_masks = []

    all_probs1 = []
    all_lbls1 = []

    for i, (train_idx, test_idx) in enumerate(kf.split(ag)):
        print("Fold: ", i + 1)

        lengths_train = [lengths[i] for i in train_idx]
        lengths_test = [lengths[i] for i in test_idx]

        print("len(train_idx",len(train_idx))

        train_idx = torch.from_numpy(train_idx)
        test_idx = torch.from_numpy(test_idx)

        ag_train = index_select(ag, 0, train_idx)
        lbls_train = index_select(lbls, 0, train_idx)
        masks_train = index_select(masks, 0, train_idx)

        ag_test = Variable(index_select(ag, 0, test_idx))
        lbls_test = Variable(index_select(lbls, 0, test_idx))
        masks_test = Variable(index_select(masks, 0, test_idx))

        code = 1
        if code ==1:
            probs_test1, lbls_test1, probs_test2, lbls_test2 = \
                epitope_run(ag_train, lbls_train, masks_train, lengths_train, weights_template, i,
                                    ag_test, lbls_test, masks_test, lengths_test)

        print("test", file=track_f)

        lbls_test2 = np.squeeze(lbls_test2)
        all_lbls2 = np.concatenate((all_lbls2, lbls_test2))
        all_lbls1.append(lbls_test1)

        probs_test_pad = torch.zeros(probs_test1.data.shape[0], MAX_AG_LENGTH, probs_test1.data.shape[2])
        probs_test_pad[:probs_test1.data.shape[0], :probs_test1.data.shape[1], :] = probs_test1.data
        probs_test_pad = Variable(probs_test_pad)

        probs_test2 = np.squeeze(probs_test2)
        #print(probs_test)
        all_probs2 = np.concatenate((all_probs2, probs_test2))
        #print(all_probs)
        #print(type(all_probs))

        all_probs1.append(probs_test_pad)

        all_masks.append(masks_test)

    lbl_mat1 = torch.cat(all_lbls1)
    prob_mat1 = torch.cat(all_probs1)
    #print("end", all_probs)
    mask_mat = torch.cat(all_masks)

    with open(output_file, "wb") as f:
        pickle.dump((lbl_mat1, prob_mat1, mask_mat, all_lbls2, all_probs2), f)

def helper_compute_metrics(matrices, aucs, mcorrs):
    matrices = np.stack(matrices)
    mean_conf = np.mean(matrices, axis=0)
    errs_conf = 2 * np.std(matrices, axis=0)

    tns = matrices[:, 0, 0]
    tps = matrices[:, 1, 1]
    fns = matrices[:, 1, 0]
    fps = matrices[:, 0, 1]

    tnsf = tns.astype(float)
    tpsf = tps.astype(float)
    fnsf = fns.astype(float)
    fpsf = fps.astype(float)

    print("tnsf", tnsf)
    print("tpsf", tpsf)
    print("fnsf", fnsf)

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
    #print("ROC AUC - original = {} +/- {}".format(auc2, auc_err2))
    # print("ROC AUC - concatenated and iterated = {} +/- {}".format(auc3, auc_err3))
    print("MCC = {} +/- {}".format(mcorr, mcorr_err))


def compute_classifier_metrics(labels, probs, labels1, probs1, threshold=0.5):
    matrices = []
    matrices1 = []

    aucs1 = []
    aucs2 = []

    mcorrs = []
    mcorrs1 = []

    all_l_pred = []

    #print("labels", labels)
    #print("probs", probs)

    for l, p in zip(labels, probs):
        #print("l", l)
        #print("p", p)
        aucs2.append(roc_auc_score(l, p))
        l_pred = (p > threshold).astype(int)
        matrices.append(confusion_matrix(l, l_pred))
        mcorrs.append(matthews_corrcoef(l, l_pred))

    for l1, p1 in zip(labels1, probs1):
        #print("in for")
        #print("l1", l1)
        #print("p1", p1)
        aucs1.append(roc_auc_score(l1, p1))
        l_pred1 = (p1 > threshold).astype(int)
        matrices1.append(confusion_matrix(l1, l_pred1))
        mcorrs1.append(matthews_corrcoef(l1, l_pred1))
        all_l_pred.append(l_pred1)

    print("all_l_pred", all_l_pred, file=track_f)

    print("Metrics with the original version")
    helper_compute_metrics(matrices=matrices,  aucs=aucs2, mcorrs =mcorrs )
    print("Metrics with probabilities concatenated")
    helper_compute_metrics(matrices=matrices1, aucs=aucs1, mcorrs=mcorrs1)


def open_crossval_results(folder="cv-ab-seq", num_results=NUM_ITERATIONS,
                          loop_filter=None, flatten_by_lengths=True):
    class_probabilities = []
    labels = []

    class_probabilities1 = []
    labels1 = []

    for r in range(num_results):
        result_filename = "{}/run-{}.p".format(folder, r)
        with open(result_filename, "rb") as f:
            lbl_mat, prob_mat, mask_mat, all_lbls, all_probs = pickle.load(f)

            lbl_mat = lbl_mat.data.cpu().numpy()
            prob_mat = prob_mat.data.cpu().numpy()
            mask_mat = mask_mat.data.cpu().numpy()

        # Get entries corresponding to the given loop
        if loop_filter is not None:
            lbl_mat = lbl_mat[loop_filter::6]
            prob_mat = prob_mat[loop_filter::6]
            mask_mat = mask_mat[loop_filter::6]

        """""
        if not flatten_by_lengths:
            class_probabilities.append(prob_mat)
            labels.append(lbl_mat)
            continue
        """

        # Discard sequence padding
        seq_lens = np.sum(np.squeeze(mask_mat), axis=1)
        seq_lens = seq_lens.astype(int)

        p = flatten_with_lengths(prob_mat, seq_lens)
        l = flatten_with_lengths(lbl_mat, seq_lens)

        class_probabilities.append(p)
        labels.append(l)

        #class_probabilities1 = np.concatenate((class_probabilities1, all_probs))
        #labels1 = np.concatenate((labels1,all_lbls))

        class_probabilities1.append(all_probs)
        labels1.append(all_lbls)

    return labels, class_probabilities, labels1, class_probabilities1

