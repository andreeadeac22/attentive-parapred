import numpy as np
np.set_printoptions(threshold=np.nan)
from torch.autograd import Variable
import torch
torch.set_printoptions(threshold=50000)
from torch import index_select

from constants import *

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

def sort_cross_batch(cdrs, cdr_masks, ag, ag_masks, ag_lbls, ag_lengths):
    order = np.argsort(ag_lengths)
    order = order.tolist()
    order.reverse()

    ag_lengths.sort(reverse=True)

    index = Variable(torch.LongTensor(order))
    if use_cuda:
        index = index.cuda()

    cdrs = torch.index_select(cdrs, 0, index)
    cdr_masks = torch.index_select(cdr_masks, 0, index)

    ag = torch.index_select(ag, 0, index)
    ag_masks = torch.index_select(ag_masks, 0, index)
    ag_lbls = torch.index_select(ag_lbls, 0, index)


    return cdrs, cdr_masks, ag, ag_masks, ag_lbls, ag_lengths

def sort_probs(probs):
    print("probs", probs)
    probs.sort()
    return probs

def vis_sort_batch(cdrs, masks, lengths, lbls):
    order = np.argsort(lengths)
    order = order.tolist()
    order.reverse()
    lengths.sort(reverse=True)
    index = Variable(torch.LongTensor(order))
    #index = torch.LongTensor(order)

    if use_cuda:
        index = index.cuda()
        cdrs = cdrs.cuda()
        lbls = lbls.cuda()
        masks = masks.cuda()

    cdrs = torch.index_select(cdrs, 0, index)
    lbls = torch.index_select(lbls, 0, index)
    masks = torch.index_select(masks, 0, index)
    return cdrs, masks, lengths, lbls, index

def ag_vis_sort_batch(cdrs, masks, lengths, lbls, ag, ag_masks):
    order = np.argsort(lengths)
    order = order.tolist()
    order.reverse()
    lengths.sort(reverse=True)
    index = Variable(torch.LongTensor(order))
    #index = torch.LongTensor(order)

    if use_cuda:
        index = index.cuda()
        cdrs = cdrs.cuda()
        lbls = lbls.cuda()
        masks = masks.cuda()

    cdrs = torch.index_select(cdrs, 0, index)
    lbls = torch.index_select(lbls, 0, index)
    masks = torch.index_select(masks, 0, index)
    ag = torch.index_select(ag, 0, index)
    ag_masks = torch.index_select(ag_masks, 0, index)
    return cdrs, masks, lengths, lbls, index, ag, ag_masks

def permute_training_cross_data(cdrs, cdr_masks, cdr_lengths, ag, ag_masks, ag_lengths, ag_lbls):
    index = torch.randperm(cdrs.shape[0])
    if use_cuda:
        index = index.cuda()

    cdrs = torch.index_select(cdrs, 0, index)
    cdr_masks = torch.index_select(cdr_masks, 0, index)
    cdr_lengths = [cdr_lengths[i] for i in index]

    ag = torch.index_select(ag, 0, index)
    ag_masks = torch.index_select(ag_masks, 0, index)
    ag_lbls = torch.index_select(ag_lbls, 0, index)
    ag_lengths = [ag_lengths[i] for i in index]

    return cdrs, cdr_masks, cdr_lengths, ag, ag_masks, ag_lengths, ag_lbls

def permute_training_data(cdrs, masks, lengths, lbls):
    index = torch.randperm(cdrs.shape[0])
    if use_cuda:
        index = index.cuda()

    cdrs = torch.index_select(cdrs, 0, index)
    lbls = torch.index_select(lbls, 0, index)
    masks = torch.index_select(masks, 0, index)
    lengths = [lengths[i] for i in index]

    return cdrs, masks, lengths, lbls

def flatten_with_lengths(matrix, lengths):
    seqs = []
    for i, example in enumerate(matrix):
        seq = example[:lengths[i]]
        seqs.append(seq)
    return np.concatenate(seqs)