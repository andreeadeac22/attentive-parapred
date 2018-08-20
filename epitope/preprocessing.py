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

import warnings
warnings.filterwarnings("ignore")

CSV_NAME = 'sabdab_27_jun_95_90.csv'
data_frame = pd.read_csv(DATA_DIRECTORY + CSV_NAME)

aa_s = "CSTPAGNDEQHRKMILVFYWU" # U for unknown

NUM_FEATURES = len(aa_s) + 7 + 6 # one-hot + extra features + chain one-hot
AG_NUM_FEATURES = len(aa_s) + 7

NUM_EXTRA_RESIDUES = 2 # The number of extra residues to include on the either side of a CDR
CONTACT_DISTANCE = 4.5 # Contact distance between atoms in Angstroms

def load_chains(csv_file):
    print("in load_chains")
    i=0
    for _, column in csv_file.iterrows():
        #if (i<15):
        #if (column['pdb'] == "4bz1"):
        pdb_name = column['pdb']
        ab_h_chain = column['Hchain']
        ab_l_chain = column['Lchain']
        antigen_chain = column['antigen_chain']
        print(pdb_name)
        print(pdb_name, file=f)
        print(ab_h_chain, file=f)
        print(ab_l_chain, file=f)
        print(antigen_chain, file=f)
        cdrs, ag, ag_names, ab_atoms = get_x_pdb_structure(PDBS_FORMAT.format(pdb_name), ab_h_chain, ab_l_chain, antigen_chain)

        ab_search = NeighbourSearch(ab_atoms)  # replace this

        yield ab_search, ag, pdb_name, cdrs
        i = i + 1

def residue_in_contact_with(res, c_search, dist=CONTACT_DISTANCE):
    print("res", res, file=track_f)
    print(sum(c_search.search(a) > 0   # search(self, centre, radius) - for each atom in res (antigen)
               for a in res.get_unpacked_list()), file=track_f)
    return any(c_search.search(a) > 0   # search(self, centre, radius) - for each atom in res (antigen)
               for a in res.get_unpacked_list())

def residue_seq_to_one(seq):
    three_to_one = lambda r: Polypeptide.three_to_one(r.name)\
        if r.name in Polypeptide.standard_aa_names else 'U'
    return list(map(three_to_one, seq))

def one_to_number(res_str):
    return [aa_s.index(r) for r in res_str]

def coords(ag_res_str):
    return [[r.x_pos, r.y_pos, r.z_pos] for r in ag_res_str]

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


def seq_to_one_hot(agc):
    ints = one_to_number(agc)
    if (len(ints) > 0):
        new_ints = torch.LongTensor(ints)
        feats = torch.Tensor(aa_features()[new_ints])
        onehot = to_categorical(ints, num_classes=len(aa_s))
        return torch.cat((onehot, feats), 1)
    else:
        return None


def process_chains_without_labels(cdrs, max_cdr_length):
    num_residues = 0
    num_in_contact = 0

    cdr_mats = []
    cdr_masks = []
    lengths = []
    for cdr_name in ["H1", "H2", "H3", "L1", "L2", "L3"]:
        # Converting residues to amino acid sequences
        #cdr_coords = coords(cdrs[cdr_name])
        cdr_chain = residue_seq_to_one(cdrs[cdr_name])
        chain_encoding = find_chain(cdr_name)
        cdr_mat = seq_to_one_hot(cdr_chain, chain_encoding)
        cdr_mat_pad = torch.zeros(max_cdr_length, NUM_FEATURES)

        if cdr_mat is not None:
            #print("cdr_mat", cdr_mat)
            cdr_mat_pad[:cdr_mat.shape[0], :] = cdr_mat
            cdr_mats.append(cdr_mat_pad)
            lengths.append(cdr_mat.shape[0])

            cdr_mask = torch.zeros(max_cdr_length, 1)
            if len(cdr_chain) > 0:
                cdr_mask[:len(cdr_chain), 0] = 1
            cdr_masks.append(cdr_mask)
        else:
            print("is None")
            print("cdrs[cdr_name]", cdrs[cdr_name])

    cdrs = torch.stack(cdr_mats)
    masks = torch.stack(cdr_masks)

    return cdrs, masks, lengths


def process_chains(ab_search, ag, max_ag_length):
    num_residues = 0
    num_in_contact = 0
    contact = {}

    #print("ag", ag)

    for ag_name, ag_chain in ag.items():
        contact[ag_name] = [residue_in_contact_with(res, ab_search) for res in ag_chain]
        #print(contact[ag_name])
        num_residues += len(contact[ag_name])
        num_in_contact += sum(contact[ag_name])

    if num_in_contact < 5:
        print("Antigen has very few contact residues: ", num_in_contact, file=f)

    ag_mats = []
    cont_mats = []
    ag_masks = []
    lengths = []
    for ag_name, _ in ag.items():
        # Converting residues to amino acid sequences
        ag_chain = residue_seq_to_one(ag[ag_name])
        #print("ag_chain", ag_chain)
        ag_mat = seq_to_one_hot(ag_chain)
        #print("ag_mat", ag_mat)
        ag_mat_pad = torch.zeros(max_ag_length, AG_NUM_FEATURES)

        if ag_mat is not None:
            #print("cdr_mat", cdr_mat)
            ag_mat_pad[:ag_mat.shape[0], :] = ag_mat
            ag_mats.append(ag_mat_pad)
            print("length", ag_mat.shape[0], file=f)
            lengths.append(ag_mat.shape[0])

            if len(contact[ag_name]) > 0:
                cont_mat = torch.FloatTensor(contact[ag_name])
                cont_mat_pad = torch.zeros(max_ag_length, 1)
                cont_mat_pad[:cont_mat.shape[0], 0] = cont_mat
            else:
                cont_mat_pad = torch.zeros(max_ag_length, 1)
            cont_mats.append(cont_mat_pad)

            ag_mask = torch.zeros(max_ag_length, 1)
            if len(ag_chain) > 0:
                ag_mask[:len(ag_chain), 0] = 1
            ag_masks.append(ag_mask)
        else:
            print("is None")
            print("contact[ag_name]", contact[ag_name])
            print("ag[ag_name]", ag[ag_name])
            #cdr_mats.append(cdr_mat_pad)
            #lengths.append(0)

        if ag_mat is not None and ag_mat.shape[0] == 1:
            print("Length is 1")
            print("ag_mat", ag_mat)
            print("ag[ag_name]", ag[ag_name])
            print("residue_seq_to_one(ag[ag_name])", residue_seq_to_one(ag[ag_name]))
            #print("seq_to_one_hot(cdr_chain, chain_encoding)", seq_to_one_hot(cdr_chain, chain_encoding, coords))
            print("len(ag_chain", len(ag_chain))
            print(ag_mask)

    ag = torch.stack(ag_mats)
    lbls = torch.stack(cont_mats)
    masks = torch.stack(ag_masks)

    return ag, lbls, masks, (num_residues, num_in_contact), lengths


def process_dataset(csv_file):
    #print("Preprocessing", file=f)
    num_in_contact = 0
    num_residues = 0

    all_ag = []
    all_ag_lbls = []
    all_ag_lengths = []
    all_ag_masks = []

    all_cdrs = []
    all_cdrs_masks = []
    all_cdrs_lengths = []

    for ab_search, ag, pdb, cdrs in load_chains(csv_file):
        print("Processing PDB ", pdb)

        #cdrs, lbls, masks, (numresidues, numincontact), lengths = process_chains(ag_search, cdrs, max_cdr_length = MAX_CDR_LENGTH)
        ag, ag_lbls, ag_masks, (numresidues, numincontact), ag_lengths = \
            process_chains(ab_search, ag, max_ag_length=MAX_AG_LENGTH)

        cdrs, cdr_masks, cdr_lengths = process_chains_without_labels(cdrs, max_cdr_length=MAX_CDR_LENGTH)



        print("ag_length", ag_lengths)

        print("num_residues", numresidues, file=f)
        print("num_in_contact", numincontact, file=f)

        num_in_contact += numincontact
        num_residues += numresidues

        all_ag.append(ag)
        all_ag_lbls.append(ag_lbls)
        all_ag_masks.append(ag_masks)
        all_ag_lengths.append(ag_lengths)

        all_cdrs.append(cdrs)
        all_cdrs_masks.append(cdr_masks)
        all_cdrs_lengths.append(cdr_lengths)


    print("num_residues", num_residues, file=f)
    print("num_in_contact", num_in_contact, file=f)

    ag = torch.cat(all_ag)
    ag_lbls = torch.cat(all_ag_lbls)
    ag_masks = torch.cat(all_ag_masks)

    cdrs = torch.cat(all_cdrs)
    cdr_masks = torch.cat(all_cdrs_masks)

    flat_lengths = [item for sublist in all_ag_lengths for item in sublist]
    flat_cdr_lengths = [item for sublist in all_cdrs_lengths for item in sublist]

    print("ag", ag, file=monitoring_file)
    print("ag_lbls", ag_lbls, file=monitoring_file)
    print("ag_masks", ag_masks, file=monitoring_file)
    print("ag_length", ag_lengths, file=monitoring_file)
    print("cdr", cdrs, file=monitoring_file)
    return {
        "ag": ag,
        "ag_lbls": ag_lbls,
        "ag_masks": ag_masks,
        "max_ab_len": MAX_AG_LENGTH,
        "pos_class_weight" : num_residues/num_in_contact,
        "ag_lengths": flat_lengths,
        "cdrs": cdrs,
        "cdr_masks":cdr_masks,
        "cdr_lengths":flat_cdr_lengths,
    }

def open_dataset(summary_file=data_frame, dataset_cache="processed-dataset.p"):
    if exists(dataset_cache) and isfile(dataset_cache):
        print("Precomputed dataset found, loading...")
        with open(dataset_cache, "rb") as f:
            dataset = pickle.load(f)
    else:
        print("Computing and storing the dataset...")
        dataset = process_dataset(summary_file)
        with open(dataset_cache, "wb") as f:
            pickle.dump(dataset, f, protocol=2)
    return dataset

f = open("chains.txt", "w+")