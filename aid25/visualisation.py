from __future__ import print_function, division
import os
import pickle
import torch
torch.set_printoptions(threshold=50000)
import pandas as pd
from Bio.PDB import Polypeptide
from os.path import isfile, exists
import numpy as np

from constants import *
from parsing import *
from search import *
from preprocessing import process_chains

DATA_DIRECTORY = 'data/'
PDBS_FORMAT = 'data/{}.pdb'
CSV_NAME = 'sabdab_27_jun_95_90.csv'
data_frame = pd.read_csv(DATA_DIRECTORY + CSV_NAME)

aa_s = "CSTPAGNDEQHRKMILVFYWU" # U for unknown
NUM_FEATURES = len(aa_s) + 7 + 6 # one-hot + extra features + chain one-hot

NUM_EXTRA_RESIDUES = 2 # The number of extra residues to include on the either side of a CDR
CONTACT_DISTANCE = 4.5 # Contact distance between atoms in Angstroms

chothia_cdr_def = { "L1" : (24, 34), "L2" : (50, 56), "L3" : (89, 97),
                    "H1" : (26, 32), "H2" : (52, 56), "H3" : (95, 102) }

def build_the_pdb_data(pdb_name):
    for _, column in data_frame.iterrows():
        if (column['pdb'] == pdb_name):
            pdb_name = column['pdb']
            ab_h_chain = column['Hchain']
            ab_l_chain = column['Lchain']
            antigen_chain = column['antigen_chain']
            print(pdb_name)
            print(pdb_name, file=f)
            print(ab_h_chain, file=f)
            print(ab_l_chain, file=f)
            print(antigen_chain, file=f)

            cdrs, ag_atoms = get_pdb_structure(PDBS_FORMAT.format(pdb_name), ab_h_chain, ab_l_chain, antigen_chain)

            ag_search = NeighbourSearch(ag_atoms)  # replace this

            cdrs, lbls, masks, (numresidues, numincontact), lengths = process_chains(ag_search, cdrs,
                                                                                     max_cdr_length=MAX_CDR_LENGTH)
            print("cdrs: ", cdrs, file=visualisation_file)
            print("lbls: ", lbls, file=visualisation_file)
            print("masks: ", masks, file=visualisation_file)
            print("max_cdr_len: ", MAX_CDR_LENGTH, file=visualisation_file)
            print("pos_class_weight: ", numresidues / numincontact, file=visualisation_file)

            dataset = {
                "cdrs" : cdrs,
                "lbls" : lbls,
                "masks" : masks,
                "max_cdr_len": MAX_CDR_LENGTH,
                "pos_class_weight" : numresidues/numincontact,
                "lengths": lengths
            }
            with open("visualisation-dataset.p", "wb") as f:
                pickle.dump(dataset, f, protocol=2)

def print_probabilities(vis_probs, index):
    # unsort
    vis_probs = torch.index_select(vis_probs, 0, index)



    # parse file
    # print to a different file - do not modify the original