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


def permute_training_data(cdrs, masks, lengths, lbls):
    index = torch.randperm(cdrs.shape[0])
    if use_cuda:
        index = index.cuda()

    cdrs = torch.index_select(cdrs, 0, index)
    lbls = torch.index_select(lbls, 0, index)
    masks = torch.index_select(masks, 0, index)
    lengths = [lengths[i] for i in index]

    return cdrs, masks, lengths, lbls