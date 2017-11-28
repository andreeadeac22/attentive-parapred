from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from preprocessing import open_dataset
from preprocessing import NUM_FEATURES, data_frame
from model import *
import torch.optim as optim
from torch import squeeze


def full_run(dataset="data/sabdab_27_jun_95_90.csv", out_weights="weights.h5"):
    print("main, full_run")
    cache_file = dataset.split("/")[-1] + ".p"
    dataset = open_dataset(data_frame, dataset_cache=cache_file)
    cdrs, lbls, masks = dataset["cdrs"], dataset["lbls"], dataset["masks"]

    epochs = 16

    print("cdrs shape", cdrs.shape)
    print("lbls shape", lbls.shape)
    print("masks shape", masks.shape)

    sample_weight = squeeze((lbls * 1.5 + 1) * masks)
    model = AbSeqModel()

    # create your optimizer
    optimizer1 = optim.Adam(model.parameters(), weight_decay = 0.01,
                            lr= 0.01) # l2 regularizer is weight_decay, included in optimizer
    optimizer2 = optim.Adam(model.parameters(), weight_decay=0.01,
                            lr=0.001)  # l2 regularizer is weight_decay, included in optimizer
    criterion = nn.BCELoss()

    input = Variable(cdrs)

    for i in range(epochs):
        if i<10:
            optimizer = optimizer1
        else:
            optimizer = optimizer2
        optimizer.zero_grad()  # zero the gradient buffers
        output = model(input)
        loss = criterion(output, lbls)
        loss.backward()
        optimizer.step()  # Does the update

    model.save_weights(out_weights)

def run_cv(dataset="data/sabdab_27_jun_95_90.csv",
           output_folder="cv-ab-seq",
           num_iters=10):
    cache_file = dataset.split("/")[-1] + ".p"
    dataset = open_dataset(dataset, dataset_cache=cache_file)
    model_factory = lambda: ab_seq_model(dataset["max_cdr_len"])

    makedirs(output_folder + "/weights")
    for i in range(num_iters):
        print("Crossvalidation run", i+1)
        output_file = "{}/run-{}.p".format(output_folder, i)
        weights_template = output_folder + "/weights/run-" + \
                           str(i) + "-fold-{}.h5"
        kfold_cv_eval(model_factory, dataset,
                      output_file, weights_template, seed=i)

full_run()