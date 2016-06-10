import os
import sys
from types import ModuleType

from ase import Atom, Atoms
from ase.io import write as ase_write
from ase.visualize import view
from jasp import *
import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns
sns.set_style("white")

def bp(info=None):
    """A breakpoint to view something and stop the rest of the script."""
    if isinstance(info, Atoms):
        view(info)    
    elif isinstance(info, list):
        for i in info:
            print(i)
            print("")
    else:
        print(info)

    sys.exit()


def paint_atoms(atoms, indices, sym=None, layers=None):
    """Update the chemical symbol of atoms in the list of indices."""
    if sym is not None:
        symbols = sym
    else:
        symbols = ["N", "O", "B", "F"]

    if layers is not None:
        for i in indices:
            for j, l in enumerate(layers):
                if i in l:
                    atoms[i].symbol = symbols[j % len(symbols)]
    else:
        for i in indices:
            atoms[i].symbol = symbols[0]


def make_pore(atoms, indices):
    """Delete atoms at indices to create a pore."""
    for index in sorted(indices, reverse=True):
        del atoms[index]


def status_converged(energy, time):
    print("Final structure calculation: Energy/f.u. = {:0.3f}. Calculation time: {:.0f} min.".format(energy, time/60.))


def status_inprogress():
    print("Final structure calculation: In progress.")


def status_unconverged(i):
    print("Distance: {:5.2f}. Did not converge.".format(i))


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


def spline(x, y, points=200):
    """Return x and y spline values over the same range as x."""
    from scipy.interpolate import interp1d
    spline = interp1d(x, y, kind='cubic')
    x_lin = np.linspace(x[0], x[-1], points)
    y_interp = spline(x_lin)

    return [x_lin, y_interp]


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


def print_image(path, data, fig_name=None, caption=None):
    if caption is not None:
        print('#+CAPTION: {}'.format(caption))
    if fig_name is not None:
        print('#+NAME: fig:{}'.format(fig_name))
    print(write_image(path, data))


def write_image(path, data, options=None):
    file_path = './img/' + path
    directory = file_path[:file_path.rfind('/')]
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    if isinstance(data, ModuleType):
        if data.__name__ == "matplotlib.pyplot":
            data.savefig(file_path)
    elif isinstance(data, Atoms):
        #atoms.rotate('x', np.pi/-5) # TODO: do a .copy() instead of this
        file_path += '.png'
        ase_write(file_path, data)

    else:
        print("No functionality for type = {}".format(type(data)))

    return '[[' + file_path + ']]'
