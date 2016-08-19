from ase.db import connect
import os
import shutil
import twodee as td

wd = "/home-guest/devonw/hydrogen2D/"
master = os.path.join(wd, 'database/master.db')
temp_master = os.path.join(wd, 'tmp/test_master.db')
shutil.copyfile(master, temp_master)

estimate = os.path.join(wd, 'scripts/2016-08-10/estimate_4.db')
temp_estimate = os.path.join(wd, 'tmp/test_estimate.db')
shutil.copyfile(estimate, temp_estimate)

db_master = connect(temp_master)
db_estimate = connect(temp_estimate)

num_calcs = sum([1 for d in db_master.select()])
print("Master: {}".format(num_calcs))

n = 0
for de in db_estimate.select():
    atoms = db_estimate.get_atoms(de.id)
    keys = de.key_value_pairs
    db_master.write(atoms, **keys)
    n += 1
    if n % 500 == 0:
        print("... " + str(n) + " added in total.")

num_calcs = sum([1 for d in db_master.select()])
print("Master: {}".format(num_calcs))
