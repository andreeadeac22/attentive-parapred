from torch.autograd import Variable
import torch
torch.set_printoptions(threshold=50000)
import torch.nn as nn
import torch.nn.functional as F
from preprocessing import open_dataset
from preprocessing import NUM_FEATURES, data_frame
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
from epitope_model import *

def run_cv(dataset="sabdab_27_jun_95_90.csv",
           output_folder="cv-ag-seq",
           num_iters=NUM_ITERATIONS):
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
    labels, probs, labels1, probs1 = open_crossval_results("cv-ag-seq", NUM_ITERATIONS)
    #labels_abip, probs_abip = open_crossval_results("cv-ab-seq-abip", 10)

    fig = plot_pr_curve(labels, probs, colours=("#0072CF", "#68ACE5"),
                        label="Epipred")
    #fig = plot_pr_curve(labels_abip, probs_abip, colours=("#D6083B", "#EB99A9"),
    #                    label="Parapred using ABiP data", plot_fig=fig)

    fig = plot_abip_pr(fig)
    fig.savefig("epi_pr.pdf")

    fig1 = plot_pr_curve(labels1, probs1, colours=("#0072CF", "#68ACE5"),
                        label="Epipred")

    fig1 = plot_abip_pr(fig)
    fig1.savefig("epi_pr1.pdf")

    # Computing overall classifier metrics
    print("Computing classifier metrics")
    compute_classifier_metrics(labels, probs, labels1, probs1, threshold=0.2)

run_cv()
process_cv_results()
