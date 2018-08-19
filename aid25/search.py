"""
Class NeighbourSearch is used to determine ground truth and neighbourhoods for antibody amino acids
"""
from __future__ import print_function
from scipy import spatial
import numpy as np

from parsing import Atom
from constants import *

class NeighbourSearch(object):
    def __init__(self, ag_atoms_list):
        self.ag_atoms = ag_atoms_list
        self.x_coord_list = []
        self.y_coord_list = []
        self.z_coord_list = []
        for atom in self.ag_atoms:
            self.x_coord_list.append(atom.x_coord)
            self.y_coord_list.append(atom.y_coord)
            self.z_coord_list.append(atom.z_coord)
        self.tree = spatial.KDTree(list(zip(self.x_coord_list, self.y_coord_list, self.z_coord_list)))

    def search(self, atom, distance):
        """
        Determines if atom is in range distance from the points in the KDTree.
        :param atom:
        :param distance:
        :return:
        """
        return len(self.tree.query_ball_point(atom.get_coord(), distance))