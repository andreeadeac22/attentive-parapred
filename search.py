from __future__ import print_function
from parsing import Atom

def distance_between_atoms(atom1, atom2):  # avoid sqrt by using distance^2
    return (atom2.x_coord-atom1.x_coord)^2 + (atom2.y_coord-atom1.y_coord)^2 + \
                      (atom2.z_coord-atom1.z_coord)^2

