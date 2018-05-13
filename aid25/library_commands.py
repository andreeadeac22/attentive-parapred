from __future__ import print_function

import torch
from docopt import docopt

from constants import *
from model import *
from parsing import get_pdb_structure_without_ag
from preprocessing import process_chains_without_labels, seq_to_one_hot_without_chain
from evaluation_tools import sort_batch_without_labels

def get_predictor():
    global _model
    from model import AbSeqModel
    if _model is None:
        _model = AbSeqModel()
        _model.load_state_dict(torch.load("cv-ab-seq/weights/run-0-fold-1.pth.tar"))
        if use_cuda:
            _model.cuda()
    return _model


def preprocess_cdr_seq(seqs):
    NUM_FEATURES = 28

    cdr_mats = []
    cdr_masks = []
    lengths = []

    cdr_mat = seq_to_one_hot_without_chain(seqs)
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
    return cdrs, masks, lengths



def process_single_cdr(seqs):

    for s in seqs:
        for r in s:
            if r not in aa_s:
                raise ValueError("'{}' is not an amino acid residue. "
                                 "Only {} are allowed.".format(r, aa_s))

    model = get_predictor()

    cdrs, masks, lengths = preprocess_cdr_seq(seqs)

    cdrs, masks, lengths, index = sort_batch_without_labels(cdrs, masks, list(lengths))

    unpacked_masks = masks
    packed_input = pack_padded_sequence(masks, list(lengths), batch_first=True)
    masks, _ = pad_packed_sequence(packed_input, batch_first=True)

    probs = model(cdrs, unpacked_masks, masks, list(lengths))

    for i, s in enumerate(seqs):
        print("# Parapred annotation of", s)
        for j, r in enumerate(s):
            print(r, probs[i, j])
        print("----------------------------------")



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

    probs = model(cdrs, unpacked_masks, masks, list(lengths))

    sigmoid = nn.Sigmoid()
    probs = sigmoid(probs)

    # unsort
    probs = torch.index_select(probs, 0, index)
    cdrs = torch.index_select(cdrs, 0, index)



def main():
    arguments = docopt(__doc__, version='Parapred v1.0.1')
    if arguments["pdb"]:
        if arguments["<pdb_file>"]:
            process_single_pdb(arguments["<pdb_file>"],
                               arguments["--abh"], arguments["--abl"])
    else:
        if arguments["cdr"]:
            process_single_cdr(arguments["<cdr_seq>"])


if __name__ == '__main__':
    main()