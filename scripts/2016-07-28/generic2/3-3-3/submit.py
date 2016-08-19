#!/usr/bin/env python

from ase.db import connect

db_path = "scripts/2016-07-28/test_db.db"
select = "train_set=True"

db = connect(db_path)

images = []
for d in db.select(select):
    atoms = db.get_atoms(d.id)
    del atoms.constraints
    images += [atoms]

from amp import Amp
from amp.descriptor import Behler
from amp.regression import NeuralNetwork

desc = Behler(**{'cutoff': 6.5})

reg = NeuralNetwork(**{'hiddenlayers': (3, 3, 3)})

calc = Amp(descriptor=desc,
           **{'label': 'scripts/2016-07-28/generic2/3-3-3/'})

train_args = {'cores': 1, 'force_goal': None, 'data_format': 'db', 'extend_variables': False, 'energy_goal': 10}


# Define global search
from amp import SimulatedAnnealing
gs = SimulatedAnnealing(**{'steps': 50, 'temperature': 70})
train_args['global_search'] = gs


calc.train(images=images,
           **train_args)

import os
wd = "scripts/2016-07-28/generic2/3-3-3/"
os.unlink(os.path.join(wd, 'train-log.txt'))
os.unlink(os.path.join(wd, 'trained-parameters.json'))

train_args = {'cores': 1, 'force_goal': None, 'data_format': 'db', 'extend_variables': False, 'energy_goal': 0.001}


calc.train(images=images,
           **train_args)

print("Training completed.")
print("End of script reached successfully.")