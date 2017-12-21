from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.nn.functional as F
from preprocessing import open_dataset
from preprocessing import NUM_FEATURES, data_frame
from model import *
import torch.optim as optim
from torch import squeeze
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from torch import index_select

from os import makedirs
from os.path import isfile

from evaluation import kfold_cv_eval

def full_run(dataset="data/sabdab_27_jun_95_90.csv", out_weights="weights.h5"):
    print("main, full_run")
    cache_file = dataset.split("/")[-1] + ".p"
    dataset = open_dataset(data_frame, dataset_cache=cache_file)
    cdrs, total_lbls, masks, lengths = dataset["cdrs"], dataset["lbls"], dataset["masks"], dataset["lengths"]

    epochs = 16

    print("cdrs shape", cdrs.shape)
    print("lbls shape", total_lbls.shape)
    print("masks shape", masks.shape)
    print("all_lengths", lengths)

    # sample_weight = squeeze((lbls * 1.5 + 1) * masks)
    model = AbSeqModel()

    # create your optimizer
    optimizer1 = optim.Adam(model.parameters(), weight_decay = 0.01,
                            lr= 0.01) # l2 regularizer is weight_decay, included in optimizer
    optimizer2 = optim.Adam(model.parameters(), weight_decay=0.01,
                            lr=0.001)  # l2 regularizer is weight_decay, included in optimizer
    criterion = nn.BCELoss()

    total_input = Variable(cdrs)

    if use_cuda:
        model.cuda()
        total_input = total_input.cuda()
        total_lbls = total_lbls.cuda()
        masks = masks.cuda()

    for i in range(epochs):
        if i<10:
            optimizer = optimizer1
        else:
            optimizer = optimizer2
        for j in range(0, cdrs.shape[0], 32):
            optimizer.zero_grad()  # zero the gradient buffers
            interval = [x for x in range(j, min(cdrs.shape[0],j+32))]
            interval = torch.LongTensor(interval)
            if use_cuda:
                interval = interval.cuda()
            input = index_select(total_input.data, 0, interval)
            input = Variable(input, requires_grad=True)
            print("j", j)
            print("input shape", input.data.shape)
            #print("lengths", lengths[j:j+32])
            output = model(input, lengths[j:j+32])
            lbls = index_select(total_lbls, 0, interval)
            print("lbls before pack", lbls.shape)
            lbls = Variable(lbls)

            packed_input = pack_padded_sequence(lbls, lengths[j:j+32], batch_first=True)

            lbls, _ = pad_packed_sequence(packed_input, batch_first=True)

            print("lbls after pack", lbls.data.shape)
            loss = criterion(output, lbls)
            loss.backward()
            optimizer.step()  # Does the update

    torch.save(model.state_dict(), "weights.h5")

def run_cv(dataset="sabdab_27_jun_95_90.csv",
           output_folder="cv-ab-seq",
           num_iters=10):
    cache_file = dataset.split("/")[-1] + ".p"
    dataset = open_dataset(dataset_cache=cache_file)

    #makedirs(output_folder + "/weights")
    for i in range(num_iters):
        print("Crossvalidation run", i+1)
        output_file = "{}/run-{}.p".format(output_folder, i)
        weights_template = output_folder + "/weights/run-" + \
                           str(i) + "-fold-{}.h5"
        kfold_cv_eval(dataset,
                      output_file, weights_template, seed=i)

run_cv()