from __future__ import print_function
from parsing import Atom
from scipy import spatial
import numpy as np

CONTACT_DISTANCE = 4.5 # Contact distance between atoms in Angstroms

class NeighbourSearch(object):
    def __init__(self, ab_atoms_list):
        self.ab_atoms = ab_atoms_list
        self.x_coord_list = []
        self.y_coord_list = []
        self.z_coord_list = []
        for atom in self.ab_atoms:
            self.x_coord_list.append(atom.x_coord)
            self.y_coord_list.append(atom.y_coord)
            self.z_coord_list.append(atom.z_coord)
        self.tree = spatial.KDTree(list(zip(self.x_coord_list, self.y_coord_list, self.z_coord_list)))

    def search(self, atom):
        return len(self.tree.query_ball_point(atom.get_coord(), CONTACT_DISTANCE))