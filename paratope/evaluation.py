"""
Evaluation suite
"""
from __future__ import print_function
from sklearn.model_selection import KFold

import numpy as np
import torch
torch.set_printoptions(threshold=50000)
from torch import squeeze
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from torch import index_select
from sklearn.metrics import confusion_matrix, roc_auc_score, matthews_corrcoef
import pickle

from atrous_run import *
from attentionRNN_run import *
from parapred_run import *
from antigen_run import *
from atrous_self_run import *
from rnn_run import *
from xself_run import *

def kfold_cv_eval(dataset, output_file="crossval-data.p",
                  weights_template="weights-fold-{}.h5", seed=0):
    """
    Performs 10-fold cross-vallidation
    :param dataset: contains antibody amino acids, ground truth values, antigen atoms
    :param output_file: where to print weights
    :param weights_template:
    :param seed: cv
    :return:
    """
    cdrs, lbls, masks, lengths, ag, ag_masks, ag_lengths, dist_mat = \
        dataset["cdrs"], dataset["lbls"], dataset["masks"], dataset["lengths"],\
        dataset["ag"], dataset["ag_masks"], dataset["ag_lengths"], dataset["dist_mat"]


    print("cdrs", cdrs.shape)
    print("ag", ag.shape)
    #print("lbls", lbls, file=data_file)
    #print("masks", masks, file=data_file)
    #print("lengths", lengths, file=data_file)

    kf = KFold(n_splits=NUM_SPLIT, random_state=seed, shuffle=True)

    all_lbls2 = []
    all_probs2 = []
    all_masks = []

    all_probs1 = []
    all_lbls1 = []

    for i, (train_idx, test_idx) in enumerate(kf.split(cdrs)):
        print("Fold: ", i + 1)
        #print(train_idx, )
        #print(test_idx)

        lengths_train = [lengths[i] for i in train_idx]
        lengths_test = [lengths[i] for i in test_idx]

        ag_lengths_train = [ag_lengths[i] for i in train_idx]
        ag_lengths_test = [ag_lengths[i] for i in test_idx]

        #print("train_idx", train_idx)

        print("len(train_idx",len(train_idx))

        train_idx = torch.from_numpy(train_idx)
        test_idx = torch.from_numpy(test_idx)

        cdrs_train = index_select(cdrs, 0, train_idx)
        lbls_train = index_select(lbls, 0, train_idx)
        mask_train = index_select(masks, 0, train_idx)
        ag_train = index_select(ag, 0, train_idx)
        ag_masks_train = index_select(ag_masks, 0, train_idx)
        dist_mat_train = index_select(dist_mat, 0, train_idx)

        cdrs_test = index_select(cdrs, 0, test_idx)
        lbls_test = index_select(lbls, 0, test_idx)
        mask_test = index_select(masks, 0, test_idx)
        ag_test = index_select(ag, 0, test_idx)
        ag_masks_test = index_select(ag_masks, 0, test_idx)
        dist_mat_test = index_select(dist_mat, 0, test_idx)

        code = 7
        if code ==1:
            probs_test1, lbls_test1, probs_test2, lbls_test2 = \
                simple_run(cdrs_train, lbls_train, mask_train, lengths_train, weights_template, i,
                                    cdrs_test, lbls_test, mask_test, lengths_test)
        if code == 2:
            probs_test1, lbls_test1, probs_test2, lbls_test2 = \
                attention_run(cdrs_train, lbls_train, mask_train, lengths_train, weights_template, i,
                              cdrs_test, lbls_test, mask_test, lengths_test)

        if code == 3:
            probs_test1, lbls_test1, probs_test2, lbls_test2 = \
                atrous_run(cdrs_train, lbls_train, mask_train, lengths_train, weights_template, i,
                                     cdrs_test, lbls_test, mask_test, lengths_test)

        if code == 4:
            probs_test1, lbls_test1, probs_test2, lbls_test2 = \
                antigen_run(cdrs_train, lbls_train, mask_train, lengths_train,
                            ag_train, ag_masks_train, ag_lengths_train, dist_mat_train, weights_template, i,
                           cdrs_test, lbls_test, mask_test, lengths_test,
                            ag_test, ag_masks_test, ag_lengths_test, dist_mat_test)

        if code == 5:
            probs_test1, lbls_test1, probs_test2, lbls_test2 = \
                atrous_self_run(cdrs_train, lbls_train, mask_train, lengths_train, weights_template, i,
                           cdrs_test, lbls_test, mask_test, lengths_test)

        if code == 6:
            probs_test1, lbls_test1, probs_test2, lbls_test2 = \
                rnn_run(cdrs_train, lbls_train, mask_train, lengths_train, weights_template, i,
                                    cdrs_test, lbls_test, mask_test, lengths_test)

        if code == 7:
            probs_test1, lbls_test1, probs_test2, lbls_test2 = \
                xself_run(cdrs_train, lbls_train, mask_train, lengths_train,
                            ag_train, ag_masks_train, ag_lengths_train, dist_mat_train, weights_template, i,
                            cdrs_test, lbls_test, mask_test, lengths_test,
                            ag_test, ag_masks_test, ag_lengths_test, dist_mat_test)

        print("test", file=track_f)

        lbls_test2 = np.squeeze(lbls_test2)
        all_lbls2 = np.concatenate((all_lbls2, lbls_test2))
        all_lbls1.append(lbls_test1)

        probs_test_pad = torch.zeros(probs_test1.data.shape[0], MAX_CDR_LENGTH, probs_test1.data.shape[2])
        probs_test_pad[:probs_test1.data.shape[0], :probs_test1.data.shape[1], :] = probs_test1.data

        probs_test2 = np.squeeze(probs_test2)
        #print(probs_test)
        all_probs2 = np.concatenate((all_probs2, probs_test2))
        #print(all_probs)
        #print(type(all_probs))

        all_probs1.append(probs_test_pad)

        all_masks.append(mask_test)

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

    tps = matrices[:, 1, 1]
    fns = matrices[:, 1, 0]
    fps = matrices[:, 0, 1]

    tpsf = tps.astype(float)
    fnsf = fns.astype(float)
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
    #print("ROC AUC - original = {} +/- {}".format(auc2, auc_err2))
    # print("ROC AUC - concatenated and iterated = {} +/- {}".format(auc3, auc_err3))
    print("MCC = {} +/- {}".format(mcorr, mcorr_err))


def compute_classifier_metrics(labels, probs, labels1, probs1, threshold=0.5):
    """
    Computes metric: precision, recall, mcc, f1
    :param labels: ground truth
    :param probs: predicted values
    :param labels1:
    :param probs1:
    :param threshold: binding/non-binding threshold
    :return:
    """
    matrices = []
    matrices1 = []

    aucs1 = []
    aucs2 = []

    mcorrs = []
    mcorrs1 = []

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

    print("Metrics with the original version")
    helper_compute_metrics(matrices=matrices,  aucs=aucs2, mcorrs =mcorrs)
    print("Metrics with probabilities concatenated")
    helper_compute_metrics(matrices=matrices1, aucs=aucs1, mcorrs=mcorrs1)


def initial_compute_classifier_metrics(labels, probs, threshold=0.5):
    matrices = []

    aucs = []

    mcorrs = []

    #print("labels", labels)
    #print("probs", probs)

    for l, p in zip(labels, probs):
        #print("l", l)
        #print("p", p)
        aucs.append(roc_auc_score(l, p))
        l_pred = (p > threshold).astype(int)
        matrices.append(confusion_matrix(l, l_pred))
        mcorrs.append(matthews_corrcoef(l, l_pred))

    print("Metrics with the original version")
    helper_compute_metrics(matrices=matrices,  aucs=aucs, mcorrs =mcorrs )
    print("Metrics with probabilities concatenated")


def open_crossval_results(folder="cv-ab-seq", num_results=NUM_ITERATIONS,
                          loop_filter=None, flatten_by_lengths=True):
    """
    Compute cross-validation results
    :param folder: folder to output results to
    :param num_results: how many times cv is executed
    :param loop_filter:
    :param flatten_by_lengths:
    :return:
    """
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


def initial_open_crossval_results(folder="cv-ab-seq", num_results=NUM_ITERATIONS,
                          loop_filter=None, flatten_by_lengths=True):
    class_probabilities = []
    labels = []

    class_probabilities1 = []
    labels1 = []

    for r in range(num_results):
        result_filename = "{}/run-{}.p".format(folder, r)
        with open(result_filename, "rb") as f:
            lbl_mat, prob_mat, mask_mat = pickle.load(f)

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
    return labels, class_probabilities

