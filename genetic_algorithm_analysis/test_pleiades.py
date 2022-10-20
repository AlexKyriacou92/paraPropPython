import os
import random
import sys
import numpy as np
from matplotlib import pyplot as pl
import time
import datetime
import h5py
from fitness_function import fitness_correlation
from makeSim import createMatrix
from genetic_functions import initialize_from_analytical
from pleiades_scripting import make_command, test_job, submit_job

sys.path.append('../')
import paraPropPython as ppp
from receiver import receiver as rx
from transmitter import tx_signal
from data import create_sim, create_rx_ranges, create_hdf_bscan, create_tx_signal
from data import create_transmitter_array, bscan, create_rxList

#Run Bscan for Data or pseudo-Data
#TODO: Write data formatter -> converts data into the Bscan format
#===================================
def save_profile(fname_profile, z_profile, n_profile):
    nDepths = len(z_profile)
    fout = open(fname_profile, 'w')
    for i in range(nDepths):
        z_i = z_profile[i]
        n_i = n_profile[i]
        line = str(round(z_i, 3)) + '\t' + str(round(n_i, 3)) + '\n'
        fout.write(line)
    fout.close()
    return -1


nStart = 1000
nIndividuals = 10
#data_nprof0 = np.genfromtxt('start_profiles/parallel-profile-0605_1st-pk.txt')
data_nprof0 = np.genfromtxt('start_profiles/aletsch_glacier_2.txt')
zprof_0 = data_nprof0[:,0]
nprof_0 = data_nprof0[:,1]
nDepths = len(zprof_0)
#Initialize Profiles
n_prof_start = initialize_from_analytical(nprof_0, 0.04*np.ones(len(nprof_0)), nStart)

n_prof_psuedo_data = nprof_0
n_min = 1.1
n_max = 1.8

for i in range(nDepths):
    if n_prof_psuedo_data[i] < n_min:
        n_prof_psuedo_data[i] = n_min
    elif n_prof_psuedo_data[i] > n_max:
        n_prof_psuedo_data[i] = n_max

fname_nprof_pseudo = 'test_nprof_start.txt'
save_profile(fname_nprof_pseudo, zprof_0, nprof_0)

n_prof_start = n_prof_start[1:]

random.shuffle(n_prof_start)
n_prof_initial = n_prof_start[:nIndividuals]
print(len(n_prof_initial))
for j in range(nIndividuals):
    n_prof_j = n_prof_initial[j]
    for i in range(nDepths):
        if n_prof_j[i] < n_min:
            n_prof_initial[j][i] = n_min
        elif n_prof_j[i] > n_max:
            n_prof_initial[j][i] = n_max
'''
fig = pl.figure(figsize=(8,5))
ax = fig.add_subplot(111)
for i in range(nIndividuals):
    ax.plot(zprof_0, n_prof_initial[i],'-o', label = str(i))
ax.grid()
ax.legend()
pl.show()
'''

#=======================================================================================

#Create n-matrix
fname_config = 'config_aletsch.txt'
fname_nmatrix = 'test_nmatrix.h5'
nGenerations = 10
createMatrix(fname_config=fname_config, n_prof_initial=n_prof_initial, z_profile=zprof_0,
             fname_nmatrix=fname_nmatrix, nGenerations = nGenerations)
#=========================================================================================

#Create Pseudo_Data
fname_output_pseudo = 'psuedo_data.h5'
os.system('python runSim_pseudo_data.py ' + fname_config + ' ' + fname_nprof_pseudo + ' ' + fname_output_pseudo)

#make_command(config_file, bscan_data_file, nprof_matrix_file, ii, jj):
#sim_command = make_command(fname_config, fname_output_pseudo, fname_nmatrix, 0, 0)

for i in range(nIndividuals):
    fname_shell = test_job(prefix='test', config_file=fname_config, bscan_data_file=fname_output_pseudo,
             nprof_matrix_file=fname_nmatrix, gene=0, individual=i)
    submit_job(fname_shell)
