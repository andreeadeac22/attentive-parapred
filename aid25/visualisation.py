"""
Helper functions for the interpretability studies.
Plugging binindg probabilities and attentional coefficients in the occupancy factor column of PDB files.
"""
from __future__ import print_function, division
import os
import pickle
import torch
torch.set_printoptions(threshold=50000)
import pandas as pd
from os.path import isfile, exists
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
import torch.nn as nn

from .constants import *
from .parsing import *
from .search import *
from .model import *
from .preprocessing import process_chains, process_ag_chains
from .evaluation_tools import *
from .ag_experiment import *

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

def build_the_pdb_data(pdb_name= visualisation_pdb):
    print("Called build_the_pdb_data")
    for _, column in data_frame.iterrows():
        if (column['pdb'] == pdb_name):
            pdb_name = column['pdb']
            ab_h_chain = column['Hchain']
            ab_l_chain = column['Lchain']
            antigen_chain = column['antigen_chain']
            print(pdb_name)
            print(pdb_name, file=visualisation_file)
            print(ab_h_chain)
            print(ab_l_chain)
            print(antigen_chain, file=visualisation_file)

            cdrs, ag_atoms, ag, ag_names = get_pdb_structure(PDBS_FORMAT.format(pdb_name), ab_h_chain, ab_l_chain, antigen_chain)

            ag_item = ag[ag_names[0]]
            ag = {}

            ag = {name: ag_item for name, item in cdrs.items()}

            data = {
                "ab_h_chain": ab_h_chain,
                "ab_l_chain":ab_l_chain,
                "cdrs": cdrs,
                "ag_name":ag_names[0],
                "ag":ag
            }

            with open(vis_dataset_file, "wb") as write_file:
                pickle.dump(data, write_file, protocol=2)


            ag_search = NeighbourSearch(ag_atoms)
            cdrs, lbls, masks, (numresidues, numincontact), lengths = process_chains(ag_search, cdrs,
                                                                                     max_cdr_length=MAX_CDR_LENGTH)
            ag, ag_masks, ag_length = process_ag_chains(ag, max_ag_length=MAX_AG_LENGTH)

            cdrs = Variable(torch.Tensor(cdrs))
            lbls = Variable(torch.Tensor(lbls))
            masks = Variable(torch.Tensor(masks))

            ag = Variable(torch.Tensor(ag))
            ag_masks = Variable(torch.Tensor(ag_masks))

            dataset = {
                "cdrs" : cdrs,
                "lbls" : lbls,
                "masks" : masks,
                "max_cdr_len": MAX_CDR_LENGTH,
                "pos_class_weight" : numresidues/numincontact,
                "lengths": lengths,
                "ag": ag,
                "ag_masks": ag_masks
            }
            return dataset

default_out_file_name = visualisation_pdb+"_copy.pdb"
ag_default_out_file_name = visualisation_pdb+"_ag_copy.pdb"

def divide(cdrs, probs):
    if use_cuda:
        cdrs = cdrs.cpu()
        probs = probs.cpu()
    cdrs = cdrs.data
    probs = probs.data.numpy()
    for i in range(6):
        if cdrs[i][0][28]>0:   # 28-33: one-hot encoding of chain
            probs_h1 = probs[i]
        if cdrs[i][0][29]>0:
            probs_h2 = probs[i]
        if cdrs[i][0][30]>0:
            probs_h3 = probs[i]
        if cdrs[i][0][31]>0:
            probs_l1 = probs[i]
        if cdrs[i][0][32]>0:
            probs_l2 = probs[i]
        if cdrs[i][0][33]>0:
            probs_l3 = probs[i]
    return probs_h1, probs_h2, probs_h3, probs_l1, probs_l2, probs_l3

def get_residue_numbers(vis_cdrs):
    print("vis_cdrs", vis_cdrs)
    vis_cdrs_h1_numbers = []
    for residue in vis_cdrs['H1']:
        vis_cdrs_h1_numbers.append(int(residue.full_seq_num))

    vis_cdrs_h2_numbers = []
    for residue in vis_cdrs['H2']:
        vis_cdrs_h2_numbers.append(int(residue.full_seq_num))

    vis_cdrs_h3_numbers = []
    for residue in vis_cdrs['H3']:
        vis_cdrs_h3_numbers.append(int(residue.full_seq_num))

    vis_cdrs_l1_numbers = []
    for residue in vis_cdrs['L1']:
        vis_cdrs_l1_numbers.append(int(residue.full_seq_num))

    vis_cdrs_l2_numbers = []
    for residue in vis_cdrs['L2']:
        vis_cdrs_l2_numbers.append(int(residue.full_seq_num))

    vis_cdrs_l3_numbers = []
    for residue in vis_cdrs['L3']:
        vis_cdrs_l3_numbers.append(int(residue.full_seq_num))

    return vis_cdrs_h1_numbers, vis_cdrs_h2_numbers, vis_cdrs_h3_numbers, \
           vis_cdrs_l1_numbers, vis_cdrs_l2_numbers, vis_cdrs_l3_numbers

def print_probabilities(model, model_type = "FP", out_file_name = default_out_file_name):
    #model.load_state_dict(torch.load("cv-ab-seq/weights/run-0-fold-1.pth.tar"))

    if use_cuda:
        model.cuda()

    print("writing to visualisation file")
    vis_dataset = build_the_pdb_data(visualisation_pdb)
    vis_cdrs, vis_masks, vis_lbls, vis_lengths = \
        vis_dataset["cdrs"], vis_dataset["masks"], vis_dataset["lbls"], vis_dataset["lengths"]

    vis_cdrs, vis_masks, vis_lengths, vis_lbls, vis_index = vis_sort_batch(vis_cdrs, vis_masks,
                                                                           list(vis_lengths),
                                                                           vis_lbls)

    vis_unpacked_masks = vis_masks
    vis_packed_input = pack_padded_sequence(vis_masks, list(vis_lengths), batch_first=True)
    vis_masks, _ = pad_packed_sequence(vis_packed_input, batch_first=True)

    if model_type == "FP":
        vis_probs = model(vis_cdrs, vis_unpacked_masks)
    else:
        if model_type == "P":
            vis_probs = model(vis_cdrs, vis_unpacked_masks)
        else:
            if model_type == "L":
                vis_probs = model(vis_cdrs, vis_unpacked_masks)
            else:
                vis_probs = model(vis_cdrs, vis_unpacked_masks)

    sigmoid = nn.Sigmoid()
    vis_probs = sigmoid(vis_probs)

    #unsort
    vis_probs = torch.index_select(vis_probs, 0, vis_index)
    vis_cdrs = torch.index_select(vis_cdrs, 0, vis_index)

    if exists(vis_dataset_file) and isfile(vis_dataset_file):
        with open(vis_dataset_file, "rb") as read_file:
            dataset = pickle.load(read_file)
    ab_h_chain = dataset["ab_h_chain"]
    ab_l_chain = dataset["ab_l_chain"]

    vis_cdrs_h1_numbers, vis_cdrs_h2_numbers, vis_cdrs_h3_numbers, \
    vis_cdrs_l1_numbers, vis_cdrs_l2_numbers, vis_cdrs_l3_numbers = get_residue_numbers(dataset["cdrs"])

    #print("vis_cdrs_h1_numbers", vis_cdrs_h1_numbers)
    #print("vis_cdrs_h2_numbers", vis_cdrs_h2_numbers)
    #print("vis_cdrs_h3_numbers", vis_cdrs_h3_numbers)

    #print("vis_cdrs_l1_numbers", vis_cdrs_l1_numbers)
    #print("vis_cdrs_l2_numbers", vis_cdrs_l2_numbers)
    #print("vis_cdrs_l3_numbers", vis_cdrs_l3_numbers)

    #print("vis_parapred_cdrs", vis_pcdrs)
    #print("vis_probs", vis_probs)

    probs_h1, probs_h2, probs_h3, probs_l1, probs_l2, probs_l3 = divide(vis_cdrs, vis_probs)

    atom = 0
    probs_h1_counter = -1
    probs_h2_counter = -1
    probs_h3_counter = -1
    probs_l1_counter = -1
    probs_l2_counter = -1
    probs_l3_counter = -1

    prev_h1_res = ""
    prev_h2_res = ""
    prev_h3_res = ""
    prev_l1_res = ""
    prev_l2_res = ""
    prev_l3_res = ""

    current_h1_res = ""
    current_h2_res = ""
    current_h3_res = ""
    current_l1_res = ""
    current_l2_res = ""
    current_l3_res = ""


    append_file = open(out_file_name, "w+")
    in_file = open(visualisation_pdb_file_name, 'r')

    for line in in_file:
        if line.startswith('ATOM') or line.startswith('HETATM'):
            #print("atom")
            atom+=1
            res_name = line[17:20]
            res_full_name = line[16:20]
            chain_id = line[21]
            res_seq_num = int(line[22:26])
            new_line = line
            if chain_id == ab_h_chain:
                if res_seq_num in vis_cdrs_h1_numbers:
                    current_h1_res = res_full_name
                    if prev_h1_res == current_h1_res:
                        new_line = line[0:60]
                        prob = probs_h1[probs_h1_counter]
                        prob = '%.4f' % prob
                        new_line += prob
                        new_line += line[66:79]
                        new_line += "\n"
                    else:
                        probs_h1_counter+=1
                        prev_h1_res = current_h1_res

                if res_seq_num in vis_cdrs_h2_numbers:
                    current_h2_res = res_full_name
                    if prev_h2_res == current_h2_res:
                        new_line = line[0:60]
                        prob = probs_h2[probs_h2_counter]
                        prob = '%.4f' % prob
                        new_line += prob
                        new_line += line[66:79]
                        new_line += "\n"
                    else:
                        probs_h2_counter += 1
                        prev_h2_res = current_h2_res

                if res_seq_num in vis_cdrs_h3_numbers:
                    current_h3_res = res_full_name
                    if prev_h3_res == current_h3_res:
                        new_line = line[0:60]
                        prob = probs_h3[probs_h3_counter]
                        prob = '%.4f' % prob
                        new_line += prob
                        new_line += line[66:79]
                        new_line += "\n"
                    else:
                        probs_h3_counter += 1
                        prev_h3_res = current_h3_res

            if chain_id == ab_l_chain:
                if res_seq_num in vis_cdrs_l1_numbers:
                    current_l1_res = res_full_name
                    if prev_l1_res == current_l1_res:
                        new_line = line[0:60]
                        prob = probs_l1[probs_l1_counter]
                        prob = '%.4f' % prob
                        new_line += prob
                        new_line += line[66:79]
                        new_line += "\n"
                    else:
                        probs_l1_counter += 1
                        prev_l1_res = current_l1_res

                if res_seq_num in vis_cdrs_l2_numbers:
                    current_l2_res = res_full_name
                    if prev_l2_res == current_l2_res:
                        new_line = line[0:60]
                        prob = probs_l2[probs_l2_counter]
                        prob = '%.4f' % prob
                        new_line += prob
                        new_line += line[66:79]
                        new_line += "\n"
                    else:
                        probs_l2_counter += 1
                        prev_l2_res = current_l2_res

                if res_seq_num in vis_cdrs_l3_numbers:
                    current_l3_res = res_full_name
                    if prev_l3_res == current_l3_res:
                        new_line = line[0:60]
                        prob = probs_l3[probs_l3_counter]
                        prob = '%.4f' % prob
                        new_line += prob
                        new_line += line[66:79]
                        new_line += "\n"
                    else:
                        probs_l3_counter += 1
                        prev_l3_res = current_l3_res
            if new_line == line:
                new_line = line[0:60]
                prob = 0.0
                prob = '%.4f' % prob
                new_line += prob
                new_line += line[66:79]
                new_line += "\n"
            append_file.write(new_line)
        else:
            new_line = line[0:60]
            prob = 0.0
            prob = '%.4f' % prob
            new_line += prob
            new_line += line[66:79]
            new_line += "\n"
            append_file.write(new_line)

def print_ag_weights(out_file_name = ag_default_out_file_name, model=AG()):
    print("in ag visual")
    #model.load_state_dict(torch.load("cv-ab-seq/run-0-fold-0.pth.tar"))
    model.eval()

    if use_cuda:
        model.cuda()

    print("writing to visualisation file")
    vis_dataset = build_the_pdb_data(visualisation_pdb)
    vis_cdrs, vis_masks, vis_lbls, vis_lengths, vis_ag, vis_ag_masks = \
        vis_dataset["cdrs"], vis_dataset["masks"], vis_dataset["lbls"], vis_dataset["lengths"], \
        vis_dataset["ag"], vis_dataset["ag_masks"]

    # i shouldn't have to sort anymore - no rnn/lstm
    #vis_cdrs, vis_masks, vis_lengths, vis_lbls, vis_index, vis_ag, vis_ag_masks = \
    #    ag_vis_sort_batch(vis_cdrs, vis_masks, list(vis_lengths), vis_lbls, vis_ag, vis_ag_masks)

    vis_probs, weights = model(vis_cdrs, vis_masks, vis_ag, vis_ag_masks)


    sigmoid = nn.Sigmoid()
    vis_probs = sigmoid(vis_probs)

    # unsort
    #vis_probs = torch.index_select(vis_probs, 0, vis_index)
    #vis_cdrs = torch.index_select(vis_cdrs, 0, vis_index)

    print("Vis probs", file=track_f)
    print("vis_probs", vis_probs, file=track_f)
    print("vis_weights", weights[1][2], file=track_f)

    values, indices = vis_probs.max(0)
    values1, pos2 = values.max(0)
    pos2 = pos2[0]
    pos1 = indices[pos2]
    pos1 = pos1[0]

    print("pos1", pos1)
    print("pos2", pos2)

    weights = weights[pos1][0]
    weights = weights[pos2][0]

    if exists(vis_dataset_file) and isfile(vis_dataset_file):
        with open(vis_dataset_file, "rb") as read_file:
            dataset = pickle.load(read_file)
    ab_h_chain = dataset["ab_h_chain"]
    ab_l_chain = dataset["ab_l_chain"]
    ag_chain = dataset["ag_name"]

    print("ab_chain", ab_h_chain)
    print("ag_chain", ag_chain)

    vis_cdrs_h1_numbers, vis_cdrs_h2_numbers, vis_cdrs_h3_numbers, \
    vis_cdrs_l1_numbers, vis_cdrs_l2_numbers, vis_cdrs_l3_numbers = get_residue_numbers(dataset["cdrs"])

    print("vis_cdrs_h1_numbers", vis_cdrs_h1_numbers)
    print("vis_cdrs_h2_numbers", vis_cdrs_h2_numbers)
    print("vis_cdrs_h3_numbers", vis_cdrs_h3_numbers)

    print("vis_cdrs_l1_numbers", vis_cdrs_l1_numbers)
    print("vis_cdrs_l2_numbers", vis_cdrs_l2_numbers)
    print("vis_cdrs_l3_numbers", vis_cdrs_l3_numbers)

    #print("vis_parapred_cdrs", vis_pcdrs)
    # print("vis_probs", vis_probs)

    probs_h1, probs_h2, probs_h3, probs_l1, probs_l2, probs_l3 = divide(vis_cdrs, vis_probs)

    atom = 0
    probs_counter = -1
    probs_ag_counter = -1

    prev_res = ""
    prev_ag_res = ""

    append_file = open(out_file_name, "w+")
    in_file = open(visualisation_pdb_file_name, 'r')

    for line in in_file:
        if line.startswith('ATOM') or line.startswith('HETATM'):
            # print("atom")
            atom += 1
            res_name = line[17:20]
            res_full_name = line[16:20]
            chain_id = line[21]
            res_seq_num = int(line[22:26])
            new_line = line
            """
            if chain_id == ab_h_chain:
                if res_seq_num in vis_cdrs_h1_numbers:
                    current_res = res_full_name
                    if prev_res == current_res:
                        if probs_counter == pos2:
                            new_line = line[0:60]
                            prob = 1.00
                            prob = '%.4f' % prob
                            new_line += prob
                            new_line += line[66:79]
                            new_line += "\n"
                    else:
                        probs_counter += 1
                        prev_res = current_res
            """
            if chain_id == ag_chain:
                #print("ag_chain")
                current_ag_res = res_full_name
                if prev_ag_res == current_ag_res:
                    new_line = line[0:60]
                    #print("pos1", pos1[0])
                    #print("pos2", pos2[0])
                    #print("probs_counter", probs_ag_counter)

                    #print(weights)
                    prob = weights[probs_ag_counter] # this needs to be weight[1][2]
                    prob = '%.4f' % prob
                    new_line += prob
                    new_line += line[66:79]
                    new_line += "\n"
                else:
                    probs_ag_counter += 1
                    prev_ag_res = current_ag_res

            if new_line == line:
                new_line = line[0:60]
                prob = 0.0
                prob = '%.4f' % prob
                new_line += prob
                new_line += line[66:79]
                new_line += "\n"
            append_file.write(new_line)
        else:
            new_line = line[0:60]
            prob = 0.0
            prob = '%.4f' % prob
            new_line += prob
            new_line += line[66:79]
            new_line += "\n"
            append_file.write(new_line)

#print_ag_weights()
