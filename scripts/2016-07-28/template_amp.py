#!/usr/bin/env python
from amp import Amp
from amp.descriptor import Behler
from amp.regression import NeuralNetwork
from ase.db import connect
import os

db_path = "{{ db_path }}"
nn_dir = "{{ nn_dir }}"
nn_dir = os.path.join(nn_dir, "")
select = "{{ select }}"
cutoff = {{ cutoff }}
h_layers = {{ h_layers }}

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
