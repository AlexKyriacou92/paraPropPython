import datetime
import os
import random
import sys
from math import pi

import subprocess
import time
import configparser

import h5py
import numpy as np
from scipy.interpolate import interp1d
from matplotlib import pyplot as pl
from genetic_algorithm import GA, read_from_config
from makeSim_nmatrix import createMatrix, createMatrix2
import sys

from genetic_functions import initialize_from_analytical, roulette, initalize_from_fluctuations, initialize
from pleiades_scripting import make_command, test_job, submit_job, test_job_data, make_command_data, make_job, countjobs
from selection_functions import selection
from genetic_functions import create_profile

sys.path.append('../')
import util
from util import get_profile_from_file, smooth_padding, do_interpolation_same_depth
from util import save_profile_to_txtfile

fname_nmatrix = sys.argv[1]
nGen = int(sys.argv[2])

with h5py.File(fname_nmatrix, 'r') as nmatrix_hdf:
    S_arr = np.array(nmatrix_hdf['S_arr'])
    S_list = S_arr[nGen]
    n_profile_matrix = np.array(nmatrix_hdf['n_profile_matrix'])
    n_profile_ii = n_profile_matrix[nGen]

nIndividuals = len(S_list)
S_list_non_zero = [x for x in S_list if x > 0]
nReal = len(S_list_non_zero)
frac = float(nReal)/float(nIndividuals)
print('ii_gen: ', nGen, 'fraction not zero, f = ', round(frac*100, 2), '%')
print(max(S_list_non_zero))
jj_ind = np.argmax(S_list_non_zero)
print(n_profile_ii[jj_ind])