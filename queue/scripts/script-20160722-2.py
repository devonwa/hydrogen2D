#!/usr/bin/env python
import twodee as td

#atoms = td.create_base('graphene', layers=1, size=3)
cans = td.candidates(mat='graphene', layers=1, size=3, pores=None, silent=False)
