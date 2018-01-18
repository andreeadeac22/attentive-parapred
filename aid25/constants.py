import torch

MAX_CDR_LENGTH = 32
use_cuda = torch.cuda.is_available()

NUM_EXTRA_RESIDUES = 2 # The number of extra residues to include on the either side of a CDR
chothia_cdr_def = { "L1" : (24, 34), "L2" : (50, 56), "L3" : (89, 97),
                    "H1" : (26, 32), "H2" : (52, 56), "H3" : (95, 102) }
cdr_names = ["H1", "H2", "H3", "L1", "L2", "L3"]

NUM_ITERATIONS = 1
NUM_SPLIT = 3

epochs = 16

track_f = open("track_file.txt", "w")

print_file = open("open_cv.txt", "w")
prob_file = open("prob_file.txt", "w")
data_file = open("dataset.txt", "w")

monitoring_file = open("monitor.txt", "w")

indices_file = open("indices.txt", "w")

sort_file = open("sort_file.txt", "w")