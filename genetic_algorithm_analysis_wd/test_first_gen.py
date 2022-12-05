import datetime
import os
import random
import h5py
import numpy as np
from scipy.interpolate import interp1d
import subprocess
import time
from math import pi

from genetic_algorithm import GA, read_from_config
from makeSim_nmatrix import createMatrix
import sys
sys.path.append('../genetic_algorithm_analysis/')
from genetic_functions import initialize_from_analytical, roulette, initalize_from_fluctuations
from pleiades_scripting import make_command, test_job, submit_job, test_job_data, make_command_data, make_job
from selection_functions import selection

fname_nmatrix = 'test_nmatrix_data.h5'
hdf_nmatrix = h5py.File(fname_nmatrix, 'r')
S_arr = np.array(hdf_nmatrix.get('S_arr'))
n_matrix = np.array(hdf_nmatrix.get('n_profile_matrix'))
print(n_matrix[0])
hdf_nmatrix.close()

fname_config = 'config_aletsch_GA.txt'
fname_psuedo_data = 'pseudo_data.h5'
nIndividuals = len(n_matrix)
for i in range(nIndividuals):
    cmd = 'python runSim_nProfile_pseudodata.py ' + fname_config + ' ' + fname_psuedo_data + ' ' + fname_nmatrix + ' ' + str(0) + ' ' + str(i)
    os.system(cmd)