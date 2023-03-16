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
def calculate_sij(n_prof_sim, n_prof_data):
    Nsim = len(n_prof_sim)
    Ndata = len(n_prof_data)

    dn = 0
    for i in range(Ndata):
        dn += (abs(n_prof_sim[i] - n_prof_data[i]) / n_prof_data[i])**2
    dn /= float(Nsim * Ndata)
    s = 1/dn
    return s
if len(sys.argv) == 2:
    path2dir = sys.argv[1]
    nOutput = 10
elif len(sys.argv) == 3:
    path2dir = sys.argv[1]
    nOutput = int(sys.argv[2])
else:
    print('error, incorrect arg number: ', len(sys.argv), ', should be 2 or 3')
    print('correct input: python ', sys.argv[0], '<path2dir> <nOutput? = 10>')
    sys.exit()
paralell_mode = True
fname_report = path2dir + 'simul_report.txt'
f_report = open(fname_report, 'r')
next(f_report)
line_2 = f_report.readline()
cols_2 = line_2.split()
foo1 = cols_2[0]
foo2 = int(cols_2[1])
next(f_report)
next(f_report)
line_3 = f_report.readline()
cols_3 = line_3.split()
print(cols_3)
fname_pseudo0 = cols_3[0]
fname_nmatrix0 = cols_3[1]
print(fname_pseudo0, fname_nmatrix0)
fname_pseudo = path2dir + fname_pseudo0
fname_nmatrix = path2dir + fname_nmatrix0
fname_config = path2dir + 'config-file.txt'


#Plot n-matrix
print('Plotting n-matrix')
hdf_nmatrix = h5py.File(fname_nmatrix, 'r')
n_profile_matrix = np.array(hdf_nmatrix.get('n_profile_matrix'))
S_nmatrix = np.array(hdf_nmatrix.get('S_arr'))
ref_data = np.array(hdf_nmatrix['reference_data'])
hdf_nmatrix.close()
if len(ref_data) == 2:
    nprof_pseudodata = ref_data[:,1]
else:
    nprof_pseudodata = ref_data

nGens0 = len(S_nmatrix)
nGens = 0 #NUmber of solved generations
nIndividuals = len(S_nmatrix[0])
for i in range(nGens0):
    if np.all(S_nmatrix == 0) == False:
        nGens += 1
S_max_arr = np.zeros(nGens)
jj_best_arr = []
for i in range(nGens):
    S_max_arr[i] = max(S_nmatrix[i])
    jj_best_arr.append(np.argmax(S_nmatrix[i]))
jj_best_arr = np.array(jj_best_arr)
ii_best = np.argmax(S_max_arr)
jj_best = int(jj_best_arr[ii_best])
print(max(S_max_arr), ii_best, jj_best)
print(S_max_arr)

coord_list = []
coord_list.append([ii_best, jj_best])
ii_list = np.random.randint(low=0, high=nGens-1, size=nOutput)
if nOutput > 1:
    for i in range(nOutput-1):
        ii_gen = ii_list[i]
        M = 100
        jj_ind_list = np.random.randint(0, nIndividuals-1, M)
        for k in range(M):
            jj_ind = jj_ind_list[k]
            S_k = S_nmatrix[ii_gen, jj_ind]
            if S_k > 0:
                break
        coords = [ii_gen, jj_ind]
        coord_list.append(coords)

print(coord_list)


cmd_prefix = 'python runSim_nProfile_from_nmatrix.py '

path2rand_files = path2dir + 'random_bscans'
if os.path.isdir(path2rand_files) == False:
    os.system('mkdir ' + path2rand_files)
fname_randlist = path2rand_files + '/' + 'bscan_list.txt'
if os.path.isfile(fname_randlist) == False:
    f_rand = open(fname_randlist, 'w')
    line = 'ii_gen \t jj_ind \t S_signal \t S_nprof \t fname_bscan'
    f_rand.write(line)
else:
    f_rand = open(fname_randlist, 'a')
config = configparser.ConfigParser()
config.read(fname_config)
prefix_0 = config['INPUT']['prefix']
for i in range(nOutput):
    ii_gen = coord_list[i][0]
    jj_ind = coord_list[i][1]
    S_value = S_nmatrix[ii_gen, jj_ind]
    n_prof_rand = n_profile_matrix[ii_gen, jj_ind]
    S_nprof = calculate_sij(n_prof_rand, nprof_pseudodata)
    fname_out_suffix = 'pseudo_data_bscan-' + str(ii_gen) + '-' + str(jj_ind) + '.h5'
    fname_out = path2rand_files + '/' + fname_out_suffix
    print('Simualting output: ii_gen = ', ii_gen, ' jj_ind = ', jj_ind, ' S_data = ', S_value, ' S_n = ', S_nprof, ' fname: ', fname_out_suffix)
    line_out = str(ii_gen) + '\t' + str(jj_ind) + '\t' + str(S_value) + '\t' + str(S_nprof) + '\t' + fname_out_suffix + '\n'
    f_rand.write(line_out)
    cmd_i = cmd_prefix + ' ' + fname_config + ' ' + fname_nmatrix + ' ' + str(ii_gen) + ' ' + str(jj_ind) + ' ' + fname_out
    
    if paralell_mode == True:
        job_prefix = 'bscan-' + prefix_0 + '-'
        jobname = job_prefix + str(ii_gen) + '-' + str(jj_ind)
        sh_file = jobname + '.sh'
        out_file = path2dir + 'outfiles' + '/' + jobname + '.out'
        print(out_file)
        make_job(sh_file, out_file, jobname, cmd_i)
        submit_job(sh_file)
        os.system('rm -f ' + sh_file)
    else:
        os.system(cmd_i)
