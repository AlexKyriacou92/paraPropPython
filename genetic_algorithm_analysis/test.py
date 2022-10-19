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
from pleiades_scripting import make_command

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
data_nprof0 = np.genfromtxt('start_profiles/parallel-profile-0504_1st-pk.txt')
zprof_0 = data_nprof0[:,0]
nprof_0 = data_nprof0[:,1]

#Initialize Profiles
n_prof_start = initialize_from_analytical(nprof_0, 0.2*np.ones(len(nprof_0)), nStart)

n_prof_psuedo_data = n_prof_start[0]
fname_nprof_pseudo = 'test_nprof_start.txt'
save_profile(fname_nprof_pseudo, zprof_0, nprof_0)

n_prof_start = n_prof_start[1:]

random.shuffle(n_prof_start)
n_prof_initial = n_prof_start[:nIndividuals]
#=======================================================================================

#Create n-matrix
fname_config = 'config_aletsch.txt'
fname_nmatrix = 'test_nmatrix.h5'
nGenerations = 10
createMatrix(fname_config=fname_config, n_prof_initial=n_prof_initial, fname_nmatrix=fname_nmatrix, nGenerations = nGenerations)
#=============================================================================================================================

#Create Pseudo_Data
fname_output_pseudo = 'psuedo_data.h5'
os.system('python runSim_pseudo_data.py ' + fname_config + ' ' + fname_nprof_pseudo + ' ' + fname_output_pseudo)

#make_command(config_file, bscan_data_file, nprof_matrix_file, ii, jj):
sim_command = make_command(fname_config, fname_output_pseudo, fname_nmatrix, 0, 0)

'''
#First Generation:
os.command()
'''