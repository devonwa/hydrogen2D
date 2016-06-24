from ase import Atoms, Atom
import numpy as np
from vasp import Vasp

import devon_toolbox as dtb


## Graphene class with neighbor cutoffs

def center_layer(atoms, layer):
    """Return the position (x,y,z) of the center of a layer of atoms.

    Args:
        atoms (Atoms): Cell of atoms.
        layer (List[int]): Indices of atoms in layer."""
    cell = np.array(atoms.get_cell())
    center = (cell[0] + cell[1]) / 2
    atoms = [a for a in atoms if a.index in layer]
    center += [0, 0, np.mean([a.position[2] for a in atoms])]
    return center


def create_base(mat='graphene', layers=1, size=1):
    """Return a relaxed structure of the base material with n layers."""
    name = 'vasp/base/mat={0}/layers={1}'.format(mat, layers)
    atoms = Vasp(name).get_atoms()

    atoms = atoms.repeat([size, size, 1])
    return atoms


def edges(atoms, unitcell):
    """Return lists of indices of edge atoms in each layer.

    Args:
        atoms (Atoms): Superstructure of atoms
        unitcell (Atoms): Unitcell making up the superstructure
    """
    edges = []
    [u1, u2, u3] = unitcell.get_cell()
    [a1, a2, a3] = atoms.get_cell()
    repeats = a1[0] / u1[0]
    for u in unitcell:
        edges.append(u.index)
        for i in range(1, int(repeats)):
            pos1 = u.position + i * u1
            pos2 = u.position + i * u2
            edges.append(dtb.closest_atom(atoms, pos2))
            edges.append(dtb.closest_atom(atoms, pos1))

    return edges


def get_neighbors(atoms, index, layer, cutoff=4.0):
    """Return neighbor indices of the atom at index for a cutoff distance.

    Determines a list of neighboring atoms to the index atom. It uses a cutoff distance to determine the absolute distance away from an atom that would constitute it as a neighbor.

    Args:
        atoms (Atoms): Cell of atoms with multiple layers of a 2D material.
        index (int): atoms[index] is the atom used to find its neighbors.
        layer (List[int]): Indices of atoms in the layer of question.
        cutoff (float): If an atom's position away is less than cutoff, it is a neighbor.
        
    Returns:
        A list of indices (int) of neighbors in atoms.
    """
    neighbors = []
    pos = atoms[index].position
    layer_atoms = [a for a in atoms if a.index in layer]
    for a in layer_atoms:
        dist = np.linalg.norm(a.position - pos)
        if dist <= cutoff and dist is not 0.000:
            neighbors.append(a.index)

    return neighbors


def layers(atoms, thresh=2.0):
    """Return lists of the indices of atoms in layers, top layer first.

    Note that the threshold could cause inaccuracy after a drastic relaxation.
    """
    height = atoms.get_cell()[2][2]
    layers = []
    unaccounted = [atom for atom in atoms]
    while len(unaccounted) > 0:
        anchor = dtb.closest_atom_to_height(unaccounted, height)

        group = []
        for atom in unaccounted:
            dist = abs(anchor.position[2] - atom.position[2])
            if dist <= thresh:
                group.append(atom)

        indices = [g.index for g in group]
        layers.append(indices)
        unaccounted = [a for a in unaccounted if a.index not in indices]
    
    return layers


def set_vacuum(atoms, vacuum):
    """Center atoms in the z-direction in a cell of size vacuum.

    Centers atoms in a unitcell with space above and below of 1/2 * vacuum. Assumes the current unitcell is centered and cell length changes only in the z-direction.
    
    Args:
        atoms (Atoms): Unitcell of atoms
        vacuum (float): Height of new unitcell

    Returns:
        An Atoms object with the new cell height.
    """
    cell = atoms.get_cell()
    center_old = cell[2][2] / 2.
    center_new = vacuum / 2.
    cell[2][2] = vacuum
    atoms.set_cell(cell)

    for atom in atoms:
        atom.position[2] = center_new - (center_old - atom.position[2])
