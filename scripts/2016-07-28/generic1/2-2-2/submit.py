#!/usr/bin/env python
from amp import Amp
from ase.db import connect
import os

db_path = "scripts/2016-07-28/test_db.db"
select = "train_set=True"

db = connect(db_path)

images = []
for d in db.select(select):
    atoms = db.get_atoms(d.id)
    del atoms.constraints
    images += [atoms]


from amp.descriptor import Behler
desc = Behler(**{'cutoff': 6.5})

from amp.regression import NeuralNetwork
reg = NeuralNetwork(**{'hiddenlayers': (2, 2, 2)})

calc = Amp(descriptor=desc,
           **{'label': 'scripts/2016-07-28/generic1/2-2-2/'})

calc.train(images=images,
           **{'global_search': None, 'data_format': 'db', 'extend_variables': False, 'force_goal': None, 'energy_goal': 0.001, 'cores': 1})

print("Training completed")