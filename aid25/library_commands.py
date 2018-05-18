#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""Paratope prediction.

Usage:
    fast_parapred cdr <cdr_seq> [--chain <chain>]
    fast_parapred pdb <pdb_name> [--model <predictor>] [--abh <ab_h_chain>] [--abl <ab_l_chain>] [--ag <ag_chain>]
    fast_parapred --help

Options:
    cdr <cdr_seq>               The input should be a CDR sequence with 2 additional residues at either end.
                                The outputs will consist of a binding probability for each amino acid.
    --chain <chain>             The name of the chain. It has to be one of {H1, H2, H3, L1, L2, L4}.
    pdb <pdb_name>              Given a PDB file name and the names of the high and the low chain,
                                it replaces the temperature factor with
                                binding probabilities.
    --abh <ab_h_chain>       Name of the antibody high chain.
    --abl <ab_l_chain>       Name of the antibody low chain.
    --model                  Predictor to be used: LSTM Baseline(L), Parapred(P), Fast-Parapred(FP) or
                                    AG-Fast-Parapred (AFP).
    --ag <ag_chain>          Name of antigen chain in PDB file.
    -h --help                    Show this help.

"""

from __future__ import print_function

import torch
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
import pkg_resources
from docopt import docopt
from torch.autograd import Variable
import torch.nn as nn

from .constants import *
from .parsing import get_pdb_structure_without_ag
from .preprocessing import process_chains_without_labels, seq_to_one_hot_without_chain, seq_to_one_hot, find_chain
from .evaluation_tools import sort_batch_without_labels


_model = None

def get_predictor(id_model="FP"):
    global _model
    from .atrous_self import AtrousSelf
    from .model import AbSeqModel
    from .rnn_model import RNNModel
    from .ag_experiment import AG
    if _model is None:
        if id_model == "L":
            _model = RNNModel()
            weights = pkg_resources.resource_filename(__name__, "cv-ab-seq/rnn_weights.pth.tar")
        if id_model == "F":
            _model = AbSeqModel()
            weights = pkg_resources.resource_filename(__name__, "cv-ab-seq/parapred_weights.pth.tar")
        if id_model == "FP":
            _model = AtrousSelf()
            weights = pkg_resources.resource_filename(__name__, "cv-ab-seq/atrous_self_weights.pth.tar")
        if id_model == "AFP":
            _model = AG()
            weights = pkg_resources.resource_filename(__name__, "cv-ab-seq/ag_weights.pth.tar")

        _model.load_state_dict(torch.load(weights))
        if use_cuda:
            _model.cuda()
    return _model

def preprocess_cdr_seq(seqs, chain):
    NUM_FEATURES = 34

    cdr_mats = []
    cdr_masks = []
    lengths = []

    cdr_mat = seq_to_one_hot(seqs, chain)
    cdr_mat_pad = torch.zeros(MAX_CDR_LENGTH, NUM_FEATURES)

    if cdr_mat is not None:
        # print("cdr_mat", cdr_mat)
        cdr_mat_pad[:cdr_mat.shape[0], :] = cdr_mat
        cdr_mats.append(cdr_mat_pad)
        lengths.append(cdr_mat.shape[0])

        cdr_mask = torch.zeros(MAX_CDR_LENGTH, 1)
        if len(seqs) > 0:
            cdr_mask[:len(seqs), 0] = 1
        cdr_masks.append(cdr_mask)
    else:
        print("is None")
        print("cdrs[cdr_name]", seqs)


    cdrs = torch.stack(cdr_mats)
    masks = torch.stack(cdr_masks)

    cdrs = Variable(cdrs)
    masks = Variable(masks)

    return cdrs, masks, lengths



def process_single_cdr(seqs, chain):

    for s in seqs:
        for r in s:
            if r not in aa_s:
                raise ValueError("'{}' is not an amino acid residue. "
                                 "Only {} are allowed.".format(r, aa_s))

    model = get_predictor()

    chain = find_chain(chain)

    cdrs, masks, lengths = preprocess_cdr_seq(seqs, chain)

    cdrs, masks, lengths, index = sort_batch_without_labels(cdrs, masks, list(lengths))


    unpacked_masks = masks
    packed_input = pack_padded_sequence(masks, list(lengths), batch_first=True)
    masks, _ = pad_packed_sequence(packed_input, batch_first=True)

    probs = model(cdrs, unpacked_masks)

    sigmoid = nn.Sigmoid()
    probs = sigmoid(probs)

    probs = probs.data.numpy()

    print("Binding probabilities for the amino acids in", seqs)
    for j, r in enumerate(seqs):
        print(r, probs[0][j])
    print("----------------------------------")


def call_predictor(id_model, model, cdrs, masks, unpacked_masks, lengths):
    if id_model == "L":
        probs = model()
    if id_model == "F":
        probs = model()
    if id_model == "FP":
        probs = model()
    if id_model == "AFP":
        probs = model()
    return probs

def process_single_pdb(pdb_name, ab_h_chain, ab_l_chain):
    model = get_predictor()
    cdrs = get_pdb_structure_without_ag(PDBS_FORMAT.format(pdb_name), ab_h_chain, ab_l_chain)
    cdrs, masks, lengths = process_chains_without_labels(cdrs, max_cdr_length=MAX_CDR_LENGTH)
    cdrs = Variable(torch.Tensor(cdrs))
    masks = Variable(torch.Tensor(masks))

    cdrs, masks, lengths, index = sort_batch_without_labels(cdrs, masks, list(lengths))

    unpacked_masks = masks
    packed_input = pack_padded_sequence(masks, list(lengths), batch_first=True)
    masks, _ = pad_packed_sequence(packed_input, batch_first=True)

    probs = model(cdrs, unpacked_masks)

    sigmoid = nn.Sigmoid()
    probs = sigmoid(probs)

    # unsort
    probs = torch.index_select(probs, 0, index)
    cdrs = torch.index_select(cdrs, 0, index)

    print("Probs", probs)



def main():
    arguments = docopt(__doc__, version='Fast-Parapred v1.0')
    if arguments["pdb"]:
        if arguments["<pdb_name>"]:
            process_single_pdb(arguments["<pdb_name>"],
                               arguments["--abh"], arguments["--abl"])
    else:
        if arguments["cdr"]:
            process_single_cdr(arguments["<cdr_seq>"], arguments["--chain"])


if __name__ == '__main__':
    main()