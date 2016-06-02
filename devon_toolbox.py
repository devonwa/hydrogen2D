import os
import sys
import numpy as np

from ase import Atom, Atoms
from ase.io import write as ase_write


def status_converged(energy, time):
    print("Final structure calculation: Energy/f.u. = {:0.3f}. Calculation time: {:.0f} min.".format(energy, time/60.))


def status_inprogress():
    print("Final structure calculation: In progress.")


def status_unconverged(i):
    print("Distance: {:5.2f}. Did not converge.".format(i))


def set_vacuum(atoms, vacuum):
    """Center atoms in the z-direction in a cell of size vacuum.

    Centers a 2D material in a unitcell with space above and below of 1/2 * vacuum. Assumes the current unitcell is centered and changes only occur in the z-direction.
    """
    cell = atoms.get_cell()
    center_old = cell[2][2] / 2.
    center_new = vacuum / 2.
    cell[2][2] = vacuum
    atoms.set_cell(cell)

    for atom in atoms:
        atom.position[2] = center_new - (center_old - atom.position[2])


def print_image(name, atoms):
    print(write_image(name, atoms))


def write_image(name, atoms):
    file_path = './img/' + name + '.png'
    directory = file_path[:file_path.rfind('/')]
    if not os.path.exists(directory):
        os.makedirs(directory)
    ase_write(file_path, atoms)
    return '[[' + file_path + ']]'
