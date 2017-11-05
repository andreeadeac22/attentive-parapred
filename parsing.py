from __future__ import print_function

class Model(object):
    def __init__(self, ab_h_chain, ab_l_chain, antigen_chain):
        self.hchain = ab_h_chain
        self.lchain = ab_l_chain
        self.agchain = antigen_chain

    def get_hchain(self):
        return self.hchain

    def get_lchain(self):
        return self.lchain

    def get_agchain(self):
        return self.agchain


class Entity(object):
    def __init__(self, name, id):
        self._name_ = name
        self._id_ = id
        self.child_list = []

    def has_child(self, child_id):
        return (child_id in self.child_list)

    def get_id(self):
        return self._id_

    def add_child(self, child):
        if (not (self.has_child_id(child.get_id()))):
            self.child_list.append(child)

    def get_child_list(self):
        return self.child_list


class Chain(Entity):
    def __init__(self, name, id):
        Entity.__init__(self, name, id)


class Residue(Entity):
    def __init__(self, name, id):
        Entity.__init__(self, name, id)


class Atom(object):
    def __init__(self, line):
        atom_features = line.split(" ")
        atom_features = [f for f in atom_features if f != '' and f != '\n']
        self.serial_num = int(line[6:11])
        self.res_name = line[17:20]
        self.chain_id = line[21]
        self.res_seq_num = int(line[22:26])
        self.x_coord = float(line[30:38])
        self.y_coord = float(line[38:46])
        self.z_coord = float(line[46:54])


## get antigen_chain by:
# get field from csv - let's say X|Y;
# search in .pdb where atom && chain_id = X or hetatm && chain_id.startsWith(X)
def get_pdb_structure(pdb_file_name):
    in_file = open(pdb_file_name, 'r')
    for line in in_file:
        if line.startswith('ATOM') or line.startswith('HETATM'):
            atom = Atom(line)
            # if atom.chain_id == 'L':