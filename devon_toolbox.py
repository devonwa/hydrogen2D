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
        if all(isinstance(i, Atoms) for i in info):
            view(info)
        else:
            for i in info:
                print(i)
                print("")
    elif info is not None:
        print(info)

    sys.exit()


def center(atoms):
    """Return the position (x,y,z) of the center of the cell."""
    cell = np.array(atoms.get_cell())
    center = (cell[0] + cell[1]) / 2
    center += cell[2] / 2
    return center


def is_the_same(x, fun, *args):
    """True if the object is unchanged during the function call."""
    import copy
    y = copy.deepcopy(x)
    fun(*args)
    return x == y


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

def result(name, calc, fu=None):
    """Print a brief calculation report."""
    atoms = calc.get_atoms()
    energy = atoms.get_potential_energy()

    if energy is None:
        stat = "Inprogress."
    else:
        time = calc.get_elapsed_time()
        stat = "Energy = {:0.4f}. Calc time: {:.0f} min.".format(energy, time/60.)

    print(name + ": " + stat)

def status_converged(energy, time):
    print("Final structure calculation: Energy/f.u. = {:0.3f}. Calculation time: {:.0f} min.".format(energy, time/60.))


def status_inprogress():
    print("Final structure calculation: In progress.")


def status_unconverged(i):
    print("Distance: {:5.2f}. Did not converge.".format(i))


def spline(x, y, points=200):
    """Return x and y spline values over the same range as x."""
    from scipy.interpolate import interp1d
    spline = interp1d(x, y, kind='cubic')
    x_lin = np.linspace(x[0], x[-1], points)
    y_interp = spline(x_lin)

    return [x_lin, y_interp]


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
