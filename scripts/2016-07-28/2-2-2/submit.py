#!/usr/bin/env python
from amp import Amp
from amp.descriptor import Behler
from amp.regression import NeuralNetwork
from ase.db import connect
import os

db_path = "scripts/2016-07-28/test_db.db"
nn_dir = "scripts/2016-07-28/2-2-2"
nn_dir = os.path.join(nn_dir, "")
select = "train_set=True"
cutoff = 6.5
h_layers = (2, 2, 2)

db = connect(db_path)

images = []
for d in db.select(select):
    atoms = db.get_atoms(d.id)
    del atoms.constraints
    images += [atoms]

desc = Behler(cutoff=cutoff)
reg = NeuralNetwork(hiddenlayers=h_layers)
calc = Amp(label=nn_dir,
           descriptor=desc,
           regression=reg)

calc.train(images=images,
           data_format='db',
           cores=1,
           energy_goal=1e-3,
           force_goal=None,
           global_search=None,
           extend_variables=False)

print("Training completed")