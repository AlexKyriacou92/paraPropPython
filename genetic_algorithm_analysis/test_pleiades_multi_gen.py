import datetime
import os
import random

import h5py
import numpy as np

from makeSim import createMatrix
from genetic_functions import initialize_from_analytical, roulette
from pleiades_scripting import make_command, test_job, submit_job
import subprocess
import time

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
nIndividuals = 200
fname_start = 'start_profiles/aletsch_glacier_2.txt'
data_nprof0 = np.genfromtxt(fname_start)
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
#n_prof_initial = n_prof_start[:nIndividuals]
n_prof_initial = np.ones((nIndividuals, len(n_prof_psuedo_data)))
for i in range(nIndividuals):
    n_prof_initial[i] = n_prof_start[i]

print(len(n_prof_initial))
for j in range(nIndividuals):
    n_prof_j = n_prof_initial[j]
    for i in range(nDepths):
        if n_prof_j[i] < n_min:
            n_prof_initial[j][i] = n_min
        elif n_prof_j[i] > n_max:
            n_prof_initial[j][i] = n_max

#=======================================================================================


#Create n-matrix
fname_config = 'config_aletsch.txt'
fname_nmatrix = 'test_nmatrix.h5'
nGenerations = 60
createMatrix(fname_config=fname_config, n_prof_initial=n_prof_initial, z_profile=zprof_0,
             fname_nmatrix=fname_nmatrix, nGenerations = nGenerations)
#=========================================================================================

#Create Pseudo_Data

fname_output_pseudo = 'psuedo_data.h5'
os.system('python runSim_pseudo_data.py ' + fname_config + ' ' + fname_nprof_pseudo + ' ' + fname_output_pseudo)

for i in range(nIndividuals):
    fname_shell = test_job(prefix='test', config_file=fname_config, bscan_data_file=fname_output_pseudo,
             nprof_matrix_file=fname_nmatrix, gene=0, individual=i)
    submit_job(fname_shell)

def countjobs():
    cmd = 'squeue | grep "kyriacou" | wc -l'
    try:
        output = int(subprocess.check_output(cmd, shell=True))
    except:
        output = 0
    return output

jj = 1
nMinutes = 1
minutes_s = 60.0
t_sleep = nMinutes * minutes_s
while jj + 1 < nGenerations:
    nJobs = countjobs()
    if nJobs == 0:
        nmatrix_hdf = h5py.File(fname_nmatrix, 'r+')
        S_arr = nmatrix_hdf['S_arr']
        n_profile_matrix = nmatrix_hdf['n_profile_matrix']

        n_profile_initial = n_profile_matrix[0]
        n_profile_parents = n_profile_matrix[jj - 1]
        S_list = S_arr[jj - 1]
        print(jj - 1)

        n_profile_children = roulette(n_profile_parents, S_list, n_profile_initial)
        n_profile_matrix[jj] = n_profile_children
        nmatrix_hdf.close()

        nIndividuals = len(n_profile_children)
        for ii in range(nIndividuals):
            fname_shell = test_job(prefix='test', config_file=fname_config, bscan_data_file=fname_output_pseudo,
                                   nprof_matrix_file=fname_nmatrix, gene=jj, individual=ii)
            submit_job(fname_shell)
        jj += 1
    else:
        print('generation: ', jj-1, ',', nJobs, 'remaining, wait ', t_sleep, ' seconds')
        print(datetime.datetime.now())
        time.sleep(t_sleep)


fname_list = [fname_config, fname_start, fname_nprof_pseudo, fname_output_pseudo, fname_nmatrix]
destination_dir = '/common/home/akyriacou/paraPropPython/paraPropPython_multisim/genetic_algorithm_analysis/paraPropData/GA_test/'
account = 'akyriacou@astro22.physik.uni-wuppertal.de'
cmd = 'scp '
for i in range(len(fname_list)):
    cmd += fname_list[i] + ' '
cmd += account + ':' + destination_dir
os.system(cmd)