import os
import random
import sys

import h5py
import numpy as np

from makeSim import createMatrix
from genetic_functions import initialize_from_analytical, roulette
from pleiades_scripting import make_command, test_job, submit_job

fname_config = sys.argv[1]
fname_matrix = sys.argv[2]
ii_gen = int(sys.argv[3])


nmatrix_hdf = h5py.File(fname_matrix, 'r+')
S_arr = nmatrix_hdf['S_arr']
n_profile_matrix = nmatrix_hdf['n_profile_matrix']
print(n_profile_matrix.shape)

n_profile_initial = n_profile_matrix[0]
n_profile_parents = n_profile_matrix[ii_gen]
S_list = S_arr[ii_gen]
print(ii_gen)

n_profile_children = roulette(n_profile_parents, S_list, n_profile_initial)
print(len(n_profile_children))
n_profile_matrix[ii_gen+1] = n_profile_children
nmatrix_hdf.close()

nIndividuals = len(n_profile_children)