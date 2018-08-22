"""
Running cross-validation, processing results
"""
from torch.autograd import Variable
import torch
torch.set_printoptions(threshold=50000)
import torch.nn as nn
import torch.nn.functional as F
from preprocessing import open_dataset
from model import *
import torch.optim as optim
from torch import squeeze
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from torch import index_select
import matplotlib
matplotlib.use('Agg')

from os import makedirs
from os.path import exists

from evaluation import *
from evaluation_tools import *
from plotting import *
from visualisation import *

def full_run(dataset="data/sabdab_27_jun_95_90.csv", out_weights="weights.h5"):
    print("main, full_run")
    cache_file = dataset.split("/")[-1] + ".p"
    dataset = open_dataset(dataset_cache=cache_file)
    cdrs, total_lbls, masks, lengths = dataset["cdrs"], dataset["lbls"], dataset["masks"], dataset["lengths"]

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
           num_iters=NUM_ITERATIONS):
    """
    Running 10-fold cross-validation 10 times.
    :param dataset: inputs (preprocessed antibody, antigen)
    :param output_folder: output location
    :param num_iters: how many iterations of cross-validation
    :return: void
    """
    cache_file = dataset.split("/")[-1] + ".p"
    dataset = open_dataset(dataset_cache=cache_file)

    dir = output_folder + "/weights"
    if not exists(dir):
        makedirs(output_folder + "/weights")
    for i in range(num_iters):
        #i=0
        print("Crossvalidation run", i+1)
        output_file = "{}/run-{}.p".format(output_folder, i)
        weights_template = output_folder + "/weights/run-" + \
                           str(i) + "-fold-{}.pth.tar"
        kfold_cv_eval(dataset,
                      output_file, weights_template, seed=i)

def process_cv_results():
    """
    Plots PR curves, output computed metrics
    :return:void
    """
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = "Times New Roman"

    # Plot ROC per loop type
    fig = None
    cols = [("#D6083B", "#EB99A9"),
            ("#0072CF", "#68ACE5"),
            ("#EA7125", "#F3BD48"),
            ("#55A51C", "#AAB300"),
            ("#8F2BBC", "#AF95A3"),
            ("#00B1C1", "#91B9A4")]


    # Plot PR curves
    print("Plotting PR curves")
    labels, probs, labels1, probs1 = open_crossval_results("cv-ab-seq", NUM_ITERATIONS)

    #labels, probs = initial_open_crossval_results("parapred-cv-ab-seq", NUM_ITERATIONS)
    #selflabels, selfprobs, selflabels1, selfprobs1 = open_crossval_results("self-cv-ab-seq", NUM_ITERATIONS)
    #_,_,aglabels, agprobs = open_crossval_results("ag-cv-ab-seq", NUM_ITERATIONS)


    fig1 = plot_pr_curve(labels1, probs1, colours=("#0072CF", "#68ACE5"),label="Parapred")

    #fig1 = plot_abip_pr(fig1)
    #fig1 = plot_pr_curve(selflabels1, selfprobs1, colours=("#228B18", "#006400"), label="Fast-Parapred", plot_fig=fig1)
    #fig1 = plot_pr_curve(aglabels, agprobs, colours=("#FF8C00", "#FFA500"), label="AG-Fast-Parapred", plot_fig=fig1)
    #fig1.savefig("pr1.pdf")

    print("Printing PDB for visualisation")
    if visualisation_flag:
        print_probabilities()

    # Computing overall classifier metrics
    print("Computing classifier metrics")
    initial_compute_classifier_metrics(labels, probs, threshold=0.4913739)

#run_cv()
process_cv_results()
