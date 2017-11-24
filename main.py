from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from preprocessing import NUM_FEATURES
import torch.optim as optim


def full_run(dataset="data/sabdab_27_jun_95_90.csv", out_weights="weights.h5"):
    print("main, full_run")
    cache_file = dataset.split("/")[-1] + ".p"
    dataset = open_dataset(dataset, dataset_cache=cache_file)
    cdrs, lbls, masks = dataset["cdrs"], dataset["lbls"], dataset["masks"]

    print("cdrs shape", cdrs.shape)
    print("lbls shape", lbls.shape)
    print("masks shape", masks.shape)


    sample_weight = np.squeeze((lbls * 1.5 + 1) * masks)
    model = Net(dataset["max_cdr_len"])

    rate_schedule = lambda e: 0.001 if e >= 10 else 0.01

    # create your optimizer
    optimizer = optim.ADAM(net.parameters())

    optimizer.zero_grad()  # zero the gradient buffers
    output = net(input)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()  # Does the update

    model.fit([cdrs, np.squeeze(masks)],
              lbls, batch_size=32, epochs=16,
              sample_weight=sample_weight,
              callbacks=[LearningRateScheduler(rate_schedule)])

    model.save_weights(out_weights)
