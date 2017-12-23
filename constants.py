import torch

MAX_CDR_LENGTH = 32
use_cuda = torch.cuda.is_available()

NUM_EXTRA_RESIDUES = 2 # The number of extra residues to include on the either side of a CDR
chothia_cdr_def = { "L1" : (24, 34), "L2" : (50, 56), "L3" : (89, 97),
                    "H1" : (26, 32), "H2" : (52, 56), "H3" : (95, 102) }
cdr_names = ["H1", "H2", "H3", "L1", "L2", "L3"]
