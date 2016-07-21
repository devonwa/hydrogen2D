from ase import Atoms
from ase.neighborlist import NeighborList
import numpy as np
from vasp import Vasp


graphene_cutoff = 0.75  # Angstrom. Should make this more permanent.


def tester():
    print("tested")


def candidates(atoms, edge=None, size=None):
    """Return candidate pore indices combinations."""
    from itertools import combinations

    cans = []
    indices = [a.index for a in atoms if a.index not in edge]

    nblist = NeighborList([graphene_cutoff for i in range(len(atoms))],
                          bothways=True,
                          self_interaction=False)
    nblist.update(atoms)


    def constraint_check(pores):
        for pore in pores:
            remains = [a.index for a in atoms if a.index not in pore]
            if is_connected(nblist, remains) and is_connected(nblist, pore):
                cans.append(pore)

    if size is not None:
        pores = combinations(indices, size)
        constraint_check(pores)
    else:
        for i in range(1, len(indices)):
            pores = combinations(indices, i)
            constraint_check(pores)

    return cans


def is_connected(reference, indices, cutoff=graphene_cutoff):
    """Return True if indices in atoms are connected.

    Args:
        reference (Atoms or NeighborList): Structure containing atoms for test. The NeighborList must be constructed with bothways=True.
        indices (List[int]): Indices of the possibly connected atoms.
        cutoff (int): Radius defining neighbors in a NeighborList. Only relevent when reference is of the Atoms type.
    """
    if isinstance(reference, Atoms):
        nblist = NeighborList([cutoff for i in range(len(reference))],
                              bothways=True,
                              self_interaction=False)
        nblist.update(reference)
    else:
        nblist = reference

    if len(indices) == 0:
        return True

    connected = [indices[0]]
    for c in connected:
        neighbs = nblist.get_neighbors(c)[0]
        for n in neighbs:
            if n in indices and n not in connected:
                connected.append(n)

    return set(indices) == set(connected)


def is_connected_working(atoms, indices):
    """Return True if indices in atoms are connected."""
    layer = layers(atoms)[0]  # TODO devon: remove this in future?

    connected = [indices[0]]
    for c in connected:
        neighbs = get_neighbors(atoms, c, layer, graphene_cutoff)
        for n in neighbs:
            if n in indices and n not in connected:
                connected.append(n)

    return set(indices) == set(connected)


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
    name = 'vasp/type=base/mat={0}/layers={1}'.format(mat, layers)
    atoms = Vasp(name).get_atoms()

    atoms = atoms.repeat([size, size, 1])
    return atoms


def closest_atom(atoms, position, exclude=None):
    """Return the index of the atom closest to a position."""
    choices = [a.index for a in atoms]

    if exclude is not None:
        choices = [i for i in choices if i not in exclude]

    closest = None
    min_dist = None
    for i in choices:
        dist = np.linalg.norm(atoms[i].position - position)
        if np.absolute(dist) < min_dist or closest is None:
            min_dist = dist
            closest = i

    return closest


def closest_atom_to_height(atoms, height):
    """Return the first atom closest to height in the z-direction."""
    closest = atoms[0]
    min_dist = abs(height - closest.position[2])

    for atom in atoms:
        dist = abs(height - atom.position[2])
        if dist < min_dist:
            closest = atom
            min_dist = dist

    return closest


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
            edges.append(closest_atom(atoms, pos2))
            edges.append(closest_atom(atoms, pos1))

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
        if dist <= cutoff and a.index != index:
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
        anchor = closest_atom_to_height(unaccounted, height)

        group = []
        for atom in unaccounted:
            dist = abs(anchor.position[2] - atom.position[2])
            if dist <= thresh:
                group.append(atom)

        indices = [g.index for g in group]
        layers.append(indices)
        unaccounted = [a for a in unaccounted if a.index not in indices]
    
    return layers


def make_pore(atoms, indices):
    """Delete atoms at indices to create a pore."""
    # TODO devon: do atoms.copy() and return atoms instead of altering current list.
    for index in sorted(indices, reverse=True):
        del atoms[index]


def pore_string(pore, leading_zeros=3):
    """Return a unique name for the pore list."""
    name = ""
    format_str = "{0:0" + str(leading_zeros) + "d}"
    for p in sorted(pore):
        name += format_str.format(p)

    if name == "":
        return "0" * leading_zeros
    else:
        return name

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


def structure(atoms, layers=1, molecs=0, thresh=2.0):
    """Return lists of the indices of different structures in a unitcell.
    
    Specifically used for my 2D material transport structures to retrieve layers and molecules. Starts from height of z=0 and moves upwards. Note: A k-means algorithm could work well here, but the one currently (2016-06-06) in scipy did not work reliably here due to some randomness. It may be worthwhile to look into it more if we want to define molecules in 3D space.

    Args:
        atoms (Atoms): Unitcell of atoms
        layers (int): Number of layers
        molecs (int): Number of molecules. 0 or 1. >1 not implemented.
        thresh (float): Total height of a layer.

    Returns:
        Dict containing indices of atoms representing layers and molecules.
        
        Example:
        {'layers': [[0, 1, 4, 5], [2, 3, 6, 7]]
         'molecs': [8,9]}
    """
    structure = {}
    structure['layers'] = []
    structure['molecs'] = []
    unaccounted = [atom for atom in atoms]
    
    for layer in range(layers):
        anchor = closest_atom_to_height(unaccounted, 0)

        group = []
        for atom in unaccounted:
            dist = abs(anchor.position[2] - atom.position[2])
            if dist <= thresh:
                group.append(atom)


        indices = [g.index for g in group]
        structure['layers'].append(indices)
        unaccounted = [a for a in unaccounted if a.index not in indices]
    
    if molecs > 0:
        structure['molecs'].append([a.index for a in unaccounted])

    return structure

