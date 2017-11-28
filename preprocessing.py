from __future__ import print_function, division
import os
import pickle
import torch
import pandas as pd
from parsing import *
from search import *
from Bio.PDB import Polypeptide
from os.path import isfile

import warnings
warnings.filterwarnings("ignore")

MAX_CDR_LENGTH = 32
DATA_DIRECTORY = 'data/'
PDBS_FORMAT = 'data/{}.pdb'
CSV_NAME = 'sabdab_27_jun_95_90.csv'
data_frame = pd.read_csv(DATA_DIRECTORY + CSV_NAME)

aa_s = "CSTPAGNDEQHRKMILVFYWU" # U for unknown
NUM_FEATURES = len(aa_s) + 7 # one-hot + extra features

NUM_EXTRA_RESIDUES = 2 # The number of extra residues to include on the either side of a CDR
CONTACT_DISTANCE = 4.5 # Contact distance between atoms in Angstroms

chothia_cdr_def = { "L1" : (24, 34), "L2" : (50, 56), "L3" : (89, 97),
                    "H1" : (26, 32), "H2" : (52, 56), "H3" : (95, 102) }

def load_chains(csv_file):
    print("in load_chains", file=f)
    print("in load_chains")
    i=0
    for _, column in csv_file.iterrows():
        #if (i<150):
        #if (column['pdb'] == "4xtr"):
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

        yield ag_search, cdrs, pdb_name
        i = i + 1

def residue_in_contact_with(res, c_search, dist=CONTACT_DISTANCE):
    return any(c_search.search(a) > 0   # search(self, centre, radius) - for each atom in res (antibody)
               for a in res.get_unpacked_list())

def residue_seq_to_one(seq):
    three_to_one = lambda r: Polypeptide.three_to_one(r.name)\
        if r.name in Polypeptide.standard_aa_names else 'U'
    return list(map(three_to_one, seq))

def one_to_number(res_str):
    return [aa_s.index(r) for r in res_str]

def aa_features():
    # Meiler's features
    prop1 = [[1.77, 0.13, 2.43,  1.54,  6.35, 0.17, 0.41],
             [1.31, 0.06, 1.60, -0.04,  5.70, 0.20, 0.28],
             [3.03, 0.11, 2.60,  0.26,  5.60, 0.21, 0.36],
             [2.67, 0.00, 2.72,  0.72,  6.80, 0.13, 0.34],
             [1.28, 0.05, 1.00,  0.31,  6.11, 0.42, 0.23],
             [0.00, 0.00, 0.00,  0.00,  6.07, 0.13, 0.15],
             [1.60, 0.13, 2.95, -0.60,  6.52, 0.21, 0.22],
             [1.60, 0.11, 2.78, -0.77,  2.95, 0.25, 0.20],
             [1.56, 0.15, 3.78, -0.64,  3.09, 0.42, 0.21],
             [1.56, 0.18, 3.95, -0.22,  5.65, 0.36, 0.25],
             [2.99, 0.23, 4.66,  0.13,  7.69, 0.27, 0.30],
             [2.34, 0.29, 6.13, -1.01, 10.74, 0.36, 0.25],
             [1.89, 0.22, 4.77, -0.99,  9.99, 0.32, 0.27],
             [2.35, 0.22, 4.43,  1.23,  5.71, 0.38, 0.32],
             [4.19, 0.19, 4.00,  1.80,  6.04, 0.30, 0.45],
             [2.59, 0.19, 4.00,  1.70,  6.04, 0.39, 0.31],
             [3.67, 0.14, 3.00,  1.22,  6.02, 0.27, 0.49],
             [2.94, 0.29, 5.89,  1.79,  5.67, 0.30, 0.38],
             [2.94, 0.30, 6.47,  0.96,  5.66, 0.25, 0.41],
             [3.21, 0.41, 8.08,  2.25,  5.94, 0.32, 0.42],
             [0.00, 0.00, 0.00,  0.00,  0.00, 0.00, 0.00]]
    return torch.Tensor(prop1)

def to_categorical(y, num_classes):
    """ Converts a class vector to binary class matrix. """
    new_y = torch.LongTensor(y)
    n = new_y.size()[0]
    categorical = torch.zeros(n, num_classes)
    arangedTensor = torch.arange(0, n)
    intaranged = arangedTensor.long()
    categorical[intaranged, new_y] = 1
    return categorical


def seq_to_one_hot(res_seq_one):
    ints = one_to_number(res_seq_one)
    if(len(ints) > 0):
        new_ints = torch.LongTensor(ints)
        feats = torch.Tensor(aa_features()[new_ints])
        onehot = to_categorical(ints, num_classes=len(aa_s))
        concatenated = torch.cat((onehot, feats), 1)
        return torch.cat((onehot, feats), 1)
    else:
        return torch.zeros(1, NUM_FEATURES)

def process_chains(ag_search, cdrs, max_cdr_length):
    num_residues = 0
    num_in_contact = 0
    contact = {}

    for cdr_name, cdr_chain in cdrs.items():
        contact[cdr_name] =  [residue_in_contact_with(res, ag_search) for res in cdr_chain]
        # print("cdr_name", cdr_name, file=f)
        # print("contact[cdr_name]", contact[cdr_name], file=f)
        # print("num_residues", len(contact[cdr_name]), file=f)
        # print("num_in_contact", sum(contact[cdr_name]), file=f)
        num_residues += len(contact[cdr_name])
        num_in_contact += sum(contact[cdr_name])

    if num_in_contact < 5:
        print("Antibody has very few contact residues: ", num_in_contact, file=f)

    cdr_mats = []
    cont_mats = []
    cdr_masks = []
    for cdr_name in ["H1", "H2", "H3", "L1", "L2", "L3"]:
        # Converting residues to amino acid sequences
        cdr_chain = residue_seq_to_one(cdrs[cdr_name])
        cdr_mat = seq_to_one_hot(cdr_chain)
        cdr_mat_pad = torch.zeros(max_cdr_length, NUM_FEATURES)
        cdr_mat_pad[:cdr_mat.shape[0], :] = cdr_mat
        cdr_mats.append(cdr_mat_pad)

        if len(contact[cdr_name]) > 0:
            cont_mat = torch.FloatTensor(contact[cdr_name])
            cont_mat_pad = torch.zeros(max_cdr_length, 1)
            cont_mat_pad[:cont_mat.shape[0], 0] = cont_mat
        else:
            cont_mat_pad = torch.zeros(max_cdr_length, 1)
        cont_mats.append(cont_mat_pad)

        cdr_mask = torch.zeros(max_cdr_length, 1)
        if len(cdr_chain) > 0:
            cdr_mask[:len(cdr_chain), 0] = 1
        cdr_masks.append(cdr_mask)

    cdrs = torch.stack(cdr_mats)
    lbls = torch.stack(cont_mats)
    masks = torch.stack(cdr_masks)

    return cdrs, lbls, masks, (num_residues, num_in_contact)


def process_dataset(csv_file):
    print("Preprocessing", file=f)
    num_in_contact = 0
    num_residues = 0

    all_cdrs = []
    all_lbls = []
    all_masks = []

    for ag_search, cdrs, pdb in load_chains(csv_file):
        print("Processing PDB ", pdb, file=f)
        print("Processing PDB ", pdb)
        cdrs, lbls, masks, (numresidues, numincontact) = process_chains(ag_search, cdrs, max_cdr_length = MAX_CDR_LENGTH)

        print("masks, masks")

        num_in_contact += numincontact
        num_residues += numresidues

        all_cdrs.append(cdrs)
        all_lbls.append(lbls)
        all_masks.append(masks)

    print("num_residues", num_residues, file=f)
    print("num_in_contact", num_in_contact, file=f)

    cdrs = torch.cat(all_cdrs)
    lbls = torch.cat(all_lbls)
    masks = torch.cat(all_masks)
    print("cdrs: ", cdrs, file=f)
    print("lbls: ", lbls, file=f)
    print("masks: ", masks, file=f)
    print("max_cdr_len: ", MAX_CDR_LENGTH, file=f)
    print("pos_class_weight: ", num_residues/num_in_contact, file=f)
    return {
        "cdrs" : cdrs,
        "lbls" : lbls,
        "masks" : masks,
        "max_cdr_len": MAX_CDR_LENGTH,
        "pos_class_weight" : num_residues/num_in_contact
    }

def open_dataset(summary_file, dataset_cache="processed-dataset.p"):

    if isfile(dataset_cache):
        print("Precomputed dataset found, loading...")
        with open(dataset_cache, "rb") as f:
            dataset = pickle.load(f)
    else:
        print("Computing and storing the dataset...")
        dataset = process_dataset(summary_file)
        with open(dataset_cache, "wb") as f:
            pickle.dump(dataset, f)

    return dataset


