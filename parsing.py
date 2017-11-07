from __future__ import print_function

NUM_EXTRA_RESIDUES = 2 # The number of extra residues to include on the either side of a CDR
chothia_cdr_def = { "L1" : (24, 34), "L2" : (50, 56), "L3" : (89, 97),
                    "H1" : (26, 32), "H2" : (52, 56), "H3" : (95, 102) }
cdr_names = ["H1", "H2", "H3", "L1", "L2", "L3"]


class Entity(object):
    def __init__(self, name, id):
        self.name = name
        self._id_ = id
        self.child_list = []

    def has_child(self, child_id):
        return (child_id in self.child_list)

    def has_empty_child_list(self):
        return (self.child_list == [])

    def get_id(self):
        return self._id_

    def get_name(self):
        return self.name

    def add_child(self, child):
        if (not (self.has_child(child.get_id()))):
            self.child_list.append(child)

    def get_child_list(self):
        return self.child_list


class Chain(Entity):
    def __init__(self, name, id):
        Entity.__init__(self, name, id)

class Residue(Entity):
    def __init__(self, name, id):
        Entity.__init__(self, name, id)

    def __repr__(self):
        name = self.get_name()
        seq = self.get_id()
        full_id = (name, seq)
        return "<Residue %s resseq=%s>" % full_id

    def get_unpacked_list(self):
        return self.child_list


class Atom(object):
    def __init__(self, line):
        atom_features = line.split(" ")
        atom_features = [f for f in atom_features if f != '' and f != '\n']
        self.serial_num = int(line[6:11])
        self.name = line[12:16]
        self.res_name = line[17:20]
        self.chain_id = line[21]
        #self.full_seq_num = line[22:27]
        self.res_seq_num = int(line[22:26])
        self.x_coord = float(line[30:38])
        self.y_coord = float(line[38:46])
        self.z_coord = float(line[46:54])
        self.coord= [self.x_coord, self.y_coord, self.z_coord]

    def get_coord(self):
        return [self.x_coord, self.y_coord, self.z_coord]

    def __repr__(self):
        full_id = (self.get_id(), self.serial_num)
        return "<Atom %s %s>" % full_id

    def get_id(self):
        return self.name

class Model(object):
    def __init__(self):
        self.cdrs = {name: [] for name in cdr_names}
        self.agatoms = []

    def get_cdrs(self):
        return self.cdrs

    def get_agatoms(self):
        return self.agatoms

    def add_agatom(self, ag_atom):
        if ag_atom not in self.agatoms:
            self.agatoms.append(ag_atom)

    def cdr_list_has_res(self, res_list, res_name, res_seq_num):
        for res in res_list:
            if res.get_id() == res_seq_num and res.get_name() == res_name:
                return res
        return None

    def agatoms_list_has_atom(self, ag_atom):
        return ag_atom in self.agatoms

    def add_residue(self, res, cdr_name):
        if res not in self.cdrs[cdr_name]:
            self.cdrs[cdr_name].append(res)


"""""
 get antigen_chain by:
 get field from csv - let's say X|Y;
 search in .pdb where atom && chain_id = X or hetatm && chain_id.startsWith(X)
"""

def get_pdb_structure(pdb_file_name, ab_h_chain, ab_l_chain, ag_chain):
    in_file = open(pdb_file_name, 'r')
    model = Model()
    cdrs = model.get_cdrs()
    print("new cdrs", cdrs)
    print(ab_h_chain, ab_l_chain, ag_chain)
    for line in in_file:
        if line.startswith('ATOM') or line.startswith('HETATM'):
            atom = Atom(line)
            res_name = atom.res_name
            res_seq_num = atom.res_seq_num
            chain_id = atom.chain_id
            for cdr_name in cdrs:
                cdr_low, cdr_hi = chothia_cdr_def[cdr_name]
                cdr_range = range(-NUM_EXTRA_RESIDUES + cdr_low, cdr_hi +
                                  NUM_EXTRA_RESIDUES + 1)
                if chain_id == ab_h_chain and cdr_name.startswith('H') and res_seq_num == 100:
                    print("seq_num", atom.res_seq_num)
                    print("full_seq_num", atom.full_seq_num)

                if ((chain_id == ab_h_chain and cdr_name.startswith('H'))\
                    or (chain_id == ab_l_chain and cdr_name.startswith('L'))) \
                                and res_seq_num in cdr_range:
                    residue = model.cdr_list_has_res(cdrs[cdr_name], res_name, res_seq_num)
                    if residue == None:
                        residue = Residue(res_name, res_seq_num)
                    residue.add_child(atom)
                    model.add_residue(residue, cdr_name)
            if " | " in ag_chain:
                c1, c2 = ag_chain.split(" | ")
                if chain_id == c1 or chain_id == c2:
                    model.add_agatom(atom)
            else:
                if chain_id == ag_chain:
                    model.add_agatom(atom)
    return cdrs, model.agatoms
    print("new cdrs.items", cdrs.items())
    print("ag_chain", model.agatoms)
