import os
import random
import sys

import h5py
import numpy as np

from makeSim import createMatrix
from genetic_functions import initialize_from_analytical, roulette
from pleiades_scripting import make_command, test_job, submit_job

fname_profile = sys.argv[1]
fname_matrix = sys.argv[2]
ii_gen = int(sys.argv[3])

profile_data = np.genfromtxt(fname_profile)
zprofile_data = profile_data[:,0]
nprofile_data = profile_data[:,1]

nmatrix_hdf = h5py.File(fname_matrix, 'r')
S_arr = nmatrix_hdf['S_arr']
n_profile_matrix = nmatrix_hdf['n_profile_matrix']

n_profile_parents = n_profile_matrix[ii_gen]
S_list = S_arr[ii_gen]

n_profile_children = roulette(n_profile_parents, S_list, nprofile_data)
n_profile_matrix[ii_gen+1] = n_profile_children
nmatrix_hdf.close()