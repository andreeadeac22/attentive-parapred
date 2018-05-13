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
        cdrs, ag_atoms, ag, ag_names = get_pdb_structure(PDBS_FORMAT.format(pdb_name), ab_h_chain, ab_l_chain, antigen_chain)

        #ag_item = ag[ag_names[0]]
        #ag = {name: ag_item for name, item in cdrs.items()}

        ag_search = NeighbourSearch(ag_atoms)  # replace this

        yield ag_search, cdrs, pdb_name, ag
        i = i + 1

def residue_in_contact_with(res, c_search, dist):
    return any(c_search.search(a, dist) > 0   # search(self, centre, radius) - for each atom in res (antibody)
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


def seq_to_one_hot(res_seq_one, chain_encoding):
    ints = one_to_number(res_seq_one)
    if(len(ints) > 0):
        new_ints = torch.LongTensor(ints)
        feats = torch.Tensor(aa_features()[new_ints])
        onehot = to_categorical(ints, num_classes=len(aa_s))
        chain_encoding = torch.Tensor(chain_encoding)
        chain_encoding = chain_encoding.expand(onehot.shape[0], 6)
        concatenated = torch.cat((onehot, feats), 1)
        #coords= torch.FloatTensor(coords)
        return torch.cat((onehot, feats, chain_encoding), 1)
    else:
        return None

def ag_seq_to_one_hot(agc):
    ints = one_to_number(agc)
    if (len(ints) > 0):
        new_ints = torch.LongTensor(ints)
        feats = torch.Tensor(aa_features()[new_ints])
        onehot = to_categorical(ints, num_classes=len(aa_s))
        concatenated = torch.cat((onehot, feats), 1)
        #print("feats", feats)
        #print("coords", coords)
        #coords = torch.FloatTensor(coords)
        return torch.cat((onehot, feats), 1)
    else:
        return None

def seq_to_one_hot_without_chain(seqs):
    return ag_seq_to_one_hot(seqs)

def find_chain(cdr_name):
    if cdr_name == "H1":
        #print("H1")
        return [1, 0, 0, 0, 0, 0,]
    if cdr_name == "H2":
        #print("H2")
        return [0, 1, 0, 0, 0, 0]
    if cdr_name == "H3":
        #print("H3")
        return [0, 0, 1, 0, 0, 0]
    if cdr_name == "L1":
        #print("L1")
        return [0, 0, 0, 1, 0, 0]
    if cdr_name == "L2":
        #print("L2")
        return [0, 0, 0, 0, 1, 0]
    if cdr_name == "L3":
        #print("L3")
        return [0, 0, 0, 0, 0, 1]

def complex_process_chains(ag_search, cdrs, max_cdr_length, ag, max_ag_length):
    num_residues = 0
    num_in_contact = 0
    contact = {}

    for cdr_name, cdr_chain in cdrs.items():
        contact[cdr_name] =  [residue_in_contact_with(res, ag_search, CONTACT_DISTANCE) for res in cdr_chain]
        num_residues += len(contact[cdr_name])
        num_in_contact += sum(contact[cdr_name])

    if num_in_contact < 5:
        print("Antibody has very few contact residues: ", num_in_contact, file=f)

    cdr_mats = []
    cont_mats = []
    cdr_masks = []
    lengths = []

    ag_mats = []
    ag_masks = []
    ag_lengths = []

    all_dist_mat = []

    for ag_name, ag_chain in ag.items():
        # Converting residues to amino acid sequences
        agc = residue_seq_to_one(ag[ag_name])
        ag_mat = ag_seq_to_one_hot(agc)
        ag_mat_pad = torch.zeros(max_ag_length, AG_NUM_FEATURES)
        ag_mat_pad[:ag_mat.shape[0], :] = ag_mat
        ag_mask = torch.zeros(max_ag_length, 1)
        ag_mask[:len(agc), 0] = 1

        cdr_chain_num=0
        maxi = 0
        for cdr_name in ["H1", "H2", "H3", "L1", "L2", "L3"]:
            # Converting residues to amino acid sequences
            # cdr_coords = coords(cdrs[cdr_name])
            cdr_chain = cdrs[cdr_name]
            enc_cdr_chain = residue_seq_to_one(cdrs[cdr_name])
            chain_encoding = find_chain(cdr_name)
            cdr_mat = seq_to_one_hot(enc_cdr_chain, chain_encoding)
            cdr_mat_pad = torch.zeros(max_cdr_length, NUM_FEATURES)

            if cdr_mat is not None:
                dist_mat = torch.zeros(MAX_CDR_LENGTH, MAX_AG_LENGTH)
                ag_num = 0
                for ag_res in ag_chain:
                    ags = NeighbourSearch(ag_res.get_unpacked_list())
                    cdr_num = 0
                    for cdr_res in cdr_chain:
                        dist_mat[cdr_num][ag_num] = residue_in_contact_with(cdr_res, ags, AG_DISTANCE)
                        sum_ag_res = sum(dist_mat[cdr_num][:])
                        if sum_ag_res > maxi:
                            maxi = sum_ag_res
                        cdr_num += 1
                    ag_num += 1

                # print("cdr_mat", cdr_mat)
                cdr_mat_pad[:cdr_mat.shape[0], :] = cdr_mat
                cdr_mats.append(cdr_mat_pad)
                lengths.append(cdr_mat.shape[0])

                if len(contact[cdr_name]) > 0:
                    cont_mat = torch.FloatTensor(contact[cdr_name])
                    cont_mat_pad = torch.zeros(max_cdr_length, 1)
                    cont_mat_pad[:cont_mat.shape[0], 0] = cont_mat
                else:
                    cont_mat_pad = torch.zeros(max_cdr_length, 1)
                cont_mats.append(cont_mat_pad)

                cdr_mask = torch.zeros(max_cdr_length, 1)
                if len(enc_cdr_chain) > 0:
                    cdr_mask[:len(enc_cdr_chain), 0] = 1
                cdr_masks.append(cdr_mask)

                ag_mats.append(ag_mat_pad)
                ag_lengths.append(ag_mat.shape[0])
                ag_masks.append(ag_mask)
                cdr_chain_num +=1
                all_dist_mat.append(dist_mat)

    cdrs = torch.stack(cdr_mats)
    lbls = torch.stack(cont_mats)
    masks = torch.stack(cdr_masks)

    ag = torch.stack(ag_mats)
    ag_masks = torch.stack(ag_masks)

    dist_mat = torch.stack(all_dist_mat)

    #print("cdrs shape", cdrs.shape)
    #print("ag shape", ag.shape)
    #print("dist", dist_mat.shape)

    return cdrs, lbls, masks, (num_residues, num_in_contact), lengths, ag, ag_masks, ag_lengths, dist_mat, maxi

def process_ag_chains(ag, max_ag_length):
    ag_mats = []
    ag_masks = []
    ag_lengths = []
    for ag_name in cdr_names:
        # Converting residues to amino acid sequences
        agc = residue_seq_to_one(ag[ag_name])
        ag_mat = ag_seq_to_one_hot(agc)
        ag_mat_pad = torch.zeros(max_ag_length, AG_NUM_FEATURES)

        if ag_mat is not None:
            ag_mat_pad[:ag_mat.shape[0], :] = ag_mat
            ag_mats.append(ag_mat_pad)
            ag_lengths.append(ag_mat.shape[0])

            ag_mask = torch.zeros(max_ag_length, 1)
            if len(agc) > 0:
                ag_mask[:len(agc), 0] = 1
            ag_masks.append(ag_mask)
        else:
            print("is None")
            print("cdrs[cdr_name]", ag[ag_name])

    ag = torch.stack(ag_mats)
    masks = torch.stack(ag_masks)

    return ag, masks, ag_lengths


def process_chains(ag_search, cdrs, max_cdr_length):
    num_residues = 0
    num_in_contact = 0
    contact = {}

    for cdr_name, cdr_chain in cdrs.items():
        contact[cdr_name] =  [residue_in_contact_with(res, ag_search, CONTACT_DISTANCE) for res in cdr_chain]
        num_residues += len(contact[cdr_name])
        num_in_contact += sum(contact[cdr_name])

    if num_in_contact < 5:
        print("Antibody has very few contact residues: ", num_in_contact, file=f)

    cdr_mats = []
    cont_mats = []
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
        else:
            print("is None")
            print("contact[cdr_name]", contact[cdr_name])
            print("cdrs[cdr_name]", cdrs[cdr_name])

    cdrs = torch.stack(cdr_mats)
    lbls = torch.stack(cont_mats)
    masks = torch.stack(cdr_masks)

    return cdrs, lbls, masks, (num_residues, num_in_contact), lengths


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


def process_dataset(csv_file):
    #print("Preprocessing", file=f)
    num_in_contact = 0
    num_residues = 0

    all_cdrs = []
    all_lbls = []
    all_lengths = []
    all_masks = []

    all_ag = []
    all_ag_lengths = []
    all_ag_masks = []

    all_dist_mat = []
    all_max = 0
    for ag_search, cdrs, pdb, ag in load_chains(csv_file):
        print("Processing PDB ", pdb)

        #cdrs, lbls, masks, (numresidues, numincontact), lengths = process_chains(ag_search, cdrs, max_cdr_length = MAX_CDR_LENGTH)
        #ag, ag_masks, ag_length = process_ag_chains(ag, max_ag_length=MAX_AG_LENGTH)

        cdrs, lbls, masks, (numresidues, numincontact), lengths, ag, ag_masks, ag_length, dist_mat, maxi = \
            complex_process_chains(ag_search=ag_search, cdrs=cdrs, max_cdr_length=MAX_CDR_LENGTH,
                                   ag=ag, max_ag_length=MAX_AG_LENGTH)

        num_in_contact += numincontact
        num_residues += numresidues

        all_cdrs.append(cdrs)
        all_lbls.append(lbls)
        all_masks.append(masks)
        all_lengths.append(lengths)

        all_ag.append(ag)
        all_ag_masks.append(ag_masks)
        all_ag_lengths.append(ag_length)

        all_dist_mat.append(dist_mat)
        #print("dist mat", dist_mat.shape)
        if maxi>all_max:
            all_max = maxi
            print("all_max", all_max)
        #print("len dist", len(all_dist_mat))

    #print("num_residues", num_residues, file=f)
    #print("num_in_contact", num_in_contact, file=f)

    cdrs = torch.cat(all_cdrs)
    lbls = torch.cat(all_lbls)
    masks = torch.cat(all_masks)

    ag = torch.cat(all_ag)
    ag_masks = torch.cat(all_ag_masks)

    dist_mat = torch.cat(all_dist_mat)
    print("total dist", dist_mat.shape)

    flat_lengths = [item for sublist in all_lengths for item in sublist]
    ag_length = [item for sublist in all_ag_lengths for item in sublist]

    return {
        "cdrs" : cdrs,
        "lbls" : lbls,
        "masks" : masks,
        "max_cdr_len": MAX_CDR_LENGTH,
        "pos_class_weight" : num_residues/num_in_contact,
        "lengths": flat_lengths,
        "ag" : ag,
        "ag_masks": ag_masks,
        "ag_lengths": ag_length,
        "dist_mat": dist_mat
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

#open_dataset()