#!/usr/bin/env python
from amp import Amp
from ase.db import connect

db_path = "{{ db_path }}"
select = "{{ select }}"

db = connect(db_path)

images = []
for d in db.select(select):
    atoms = db.get_atoms(d.id)
    del atoms.constraints
    images += [atoms]

from amp.descriptor import {{ desc_type }}
desc = {{ desc_type }}(**{{ desc_args }})

from amp.regression import {{ reg_type }}
reg = {{ reg_type }}(**{{ reg_args }})

calc = Amp(descriptor=desc,
           **{{ amp_args }})

calc.train(images=images,
           **{{ train_args }})

print("Training completed")
