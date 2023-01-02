import datetime
import os
import random
import sys
from math import pi

import h5py
import numpy as np
from scipy.interpolate import interp1d

from genetic_algorithm import GA, read_from_config
from makeSim_nmatrix import createMatrix
import sys
from sys import getsizeof

from genetic_functions import initialize_from_analytical, roulette, initalize_from_fluctuations, initialize
from pleiades_scripting import make_command, test_job, submit_job, test_job_data, make_command_data, make_job
from selection_functions import selection

import subprocess
import time
import configparser

sys.path.append('../')
from data import create_sim, create_rxList_from_file, create_tx_signal, create_transmitter_array

nArgs = len(sys.argv)
print(nArgs)
if nArgs == 5:
    fname_config = sys.argv[1]
    fname_nmatrix = sys.argv[2]
    fname_output_prefix = sys.argv[3]
    nOutput = int(sys.argv[4])
    paralell_mode0 = True
elif nArgs == 6:
    fname_config = sys.argv[1]
    fname_nmatrix = sys.argv[2]
    fname_output_prefix = sys.argv[3]
    nOutput = int(sys.argv[4])
    paralell_mode0 = sys.argv[5]
print(nOutput)
yes_choices = ['yes', 'y']
no_choices = ['no', 'n']

if nOutput == 0:
    print('please enter more than one simulation to be run')
    sys.exit()
elif nOutput > 10:
    sim_0 = create_sim(fname_config)
    tx_signal = create_tx_signal(fname_config)
    rxList0 = create_rxList_from_file(fname_config)
    tx_depths = create_transmitter_array(fname_config)
    nDepths = len(tx_depths)
    nReceivers = len(rxList0)
    bscan_npy = np.zeros((nDepths, nReceivers, tx_signal.nSamples), dtype='complex')
    nBytes = getsizeof(bscan_npy)
    print('caution, you have entered ', nOutput, ' this may lead to memory overload')
    print('memory size: ', int(float(nOutput*nBytes)/1e6), ' MB')
    user_input = input('proceed? enter: (yes/y) or (no/n): ')
    if user_input.lower() in yes_choices:
        pass
    elif user_input.lower() in no_choices:
        print('exiting')
        sys.exit()
    else:
        print('invalid response, exiting')
        sys.exit()
if nArgs == 6:
    if paralell_mode0 == 'True':
        paralell_mode = True
    elif paralell_mode0 == 'False':
        paralell_mode = False
    else:
        'invalid entry, set parallel mode to True or False'
        sys.exit()
else:
    paralell_mode = paralell_mode0

print('opening nmatrix results')
hdf_results = h5py.File(fname_nmatrix, 'r')

n_profile_matrix = np.array(hdf_results.get('n_profile_matrix'))
S_results = np.array(hdf_results.get('S_arr'))
z_profile = np.array(hdf_results.get('z_profile'))

nGenerations = int(hdf_results.attrs['nGenerations'])
nIndividuals = int(hdf_results.attrs['nIndividuals'])
rxList = np.array(hdf_results.get('rxList'))
signalPulse = np.array(hdf_results.get('signalPulse'))
tspace = np.array(hdf_results.get('tspace'))
source_depths = np.array(hdf_results.get('source_depths'))

hdf_results.close()

S_best = np.ones(nGenerations)
gens = np.arange(1, nGenerations+1, 1)


best_individuals = []
'''
all_scores = []
all_indices = []
'''
for i in range(nGenerations):
    S_best[i] = max(S_results[i])
    best_individuals.append(np.argmax(S_results[i]))
    '''
    for j in range(nIndividuals):
        if S_results != 0:
            all_scores.append(S_results[i,j])
            all_indices.append([i,j])
            
    '''
M = nGenerations // nOutput
S_list_output = []
inds_output = []

for i in range(nOutput-1):
    ii = M * i
    S_ii = S_best[ii]
    S_list_output.append(S_ii)
    inds_output.append([ii,best_individuals[ii]])

'''
all_scores = np.array(all_scores)
all_indices = np.array(all_indices)
inds = all_scores.argssort()
sorted_scores = all_scores[inds]
sorted_indices = all_indices[inds]

nSelected = len(sorted_scores)
M = nSelected // nOutput
S_list_output = []
inds_output = []

for i in range(nOutput-1):
'''
ii_best_gen = np.argmax(S_best)
jj_best_ind = best_individuals[ii_best_gen]
n_profile_best = n_profile_matrix[ii_best_gen, jj_best_ind]

S_list_output.append(S_best[ii_best_gen])
inds_output.append([ii_best_gen, jj_best_ind])

fname_report = fname_output_prefix + 'simul_report.txt'
fout = open(fname_report, 'w')
fout.write('pathto\tnOutput \n' + fname_output_prefix  + '\t' + str(nOutput) + '\n' + '\n')
fout.write('gen\tind\tS\tfname_out\n')

config = configparser.ConfigParser()
config.read(fname_config)
prefix_0 = config['INPUT']['prefix']

for i in range(nOutput):
    cmd_prefix = 'python runSim_nProfile_from_nmatrix.py '
    ii_gen = inds_output[i][0]
    jj_ind = inds_output[i][1]
    print('Simualting output: ii_gen = ', ii_gen, ' jj_ind = ', jj_ind, ' S = ', S_list_output[i])
    fname_output_suffix = 'pseudo_bscan_output_' + str(ii_gen) + '_' + str(jj_ind) + '.h5'
    fname_out = fname_output_prefix + fname_output_suffix
    cmd_i = cmd_prefix + ' ' + fname_config + ' ' + fname_nmatrix + ' ' + str(ii_gen) + ' ' + str(jj_ind) + ' ' + fname_out

    line = str(ii_gen) + '\t' + str(jj_ind) + '\t' + str(S_list_output[i]) + '\t' + fname_output_suffix + '\n'
    fout.write(line)
    if paralell_mode == True:
        job_prefix = 'bscan-' + prefix_0 + '-'
        jobname = job_prefix + str(ii_gen) + '-' + str(jj_ind)
        sh_file = jobname + '.sh'
        out_file = fname_output_prefix + 'outfiles' + '/' + jobname + '.out'
        print(out_file)
        make_job(sh_file, out_file, jobname, cmd_i)
        submit_job(sh_file)
        os.system('rm -f ' + sh_file)
    else:
        os.system(cmd_i)