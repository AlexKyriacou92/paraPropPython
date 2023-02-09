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

from genetic_functions import initialize_from_analytical, roulette, initalize_from_fluctuations, initialize
from pleiades_scripting import make_command, test_job, submit_job, test_job_data, make_command_data, make_job
from selection_functions import selection

import subprocess
import time
import configparser

sys.path.append('../')
import util
#=======================================================================================================================
# Initialize the GA
#=======================================================================================================================
def countjobs():
    cmd = 'squeue | grep "kyriacou" | wc -l'
    try:
        output = int(subprocess.check_output(cmd, shell=True))
    except:
        output = 0
    return output

def get_profile_from_file(fname):
    profile_data = np.genfromtxt(fname)
    z_profile = profile_data[:,0]
    n_profile = profile_data[:,1]
    return n_profile, z_profile

def do_interpolation(zprof_in, nprof_in, zmin, zmax, N):
    f_interp = interp1d(zprof_in, nprof_in)
    zprof_out = np.linspace(zmin, zmax, N)
    nprof_out = np.ones(N)
    ii_min = util.findNearest(zprof_in, zmin)
    ii_max = util.findNearest(zprof_in, zmax)
    nprof_out[0] = nprof_in[ii_min]
    nprof_out[-1] = nprof_in[ii_max]
    nprof_out[1:-1] = f_interp(zprof_out[1:-1])
    return nprof_out, zprof_out

nRand = 10

var_range = 0.025

nprof_gul0, zprof_gul0 = get_profile_from_file('share/guliya.txt')
nDepths = 28
nprof_gul, zprof_gul = do_interpolation(zprof_gul0, nprof_gul0, 1, 15, nDepths)

fname_prof_new = 'share/guliya_reduced.txt'
if os.path.isdir(fname_prof_new) == False:
    with open(fname_prof_new,'w+') as f:
        for i in range(nDepths):
            line = str(round(zprof_gul[i],3)) + '\t' + str(round(nprof_gul[i], 3)) + '\n'
            f.write(line)

nRand = 10

max_var_list = [0.025, 0.05, 0.075, 0.1]
N = len(max_var_list)
randVec_arr = np.zeros((N, nRand, nDepths))
nprof_rand = np.zeros((N, nRand, nDepths))
for i in range(len(max_var_list)):
    for j in range(nRand):
        max_var = max_var_list[i]
        randVec_arr[i,j,:] = np.random.uniform(-max_var,max_var,nDepths)
        nprof_rand[i,j] = nprof_gul + randVec_arr[i,j,:]

results_dir = 'calculate_S'
os.system('mkdir ' + results_dir)
fname_pseudo_output = results_dir + '/' + 'guliya_los_res_pseudo.h5'
fname_nmatrix_output = results_dir + '/' + 'guliya_low_res_nmatrix.h5'

fname_config = 'config_aletsch_GA_pseudo.txt'

createMatrix(fname_config=fname_config, n_prof_initial=nprof_rand[0], z_profile=zprof_gul,
                 fname_nmatrix=fname_nmatrix_output, nGenerations=N)
cmd = 'python runSim_pseudodata_from_txt.py ' + fname_config + ' ' + fname_prof_new + ' ' + fname_pseudo_output
os.system(cmd)
cmd_prefix = 'python runSim_nProfile_pseudodata.py '
#Set up Simulation Output Files
config = configparser.ConfigParser()
config.read(fname_config)
sim_mode = config['INPUT']['sim_mode']
job_prefix = config['INPUT']['prefix']
dir_outfiles = results_dir + '/outfiles'
os.system('mkdir ' + dir_outfiles)

for i in range(N):
    for j in range(nRand):
        nmatrix_hdf = h5py.File(fname_nmatrix_output, 'r+')
        #S_arr = np.array(nmatrix_hdf['S_arr'])
        n_profile_matrix = nmatrix_hdf['n_profile_matrix']
        n_profile_matrix[j] = nprof_rand[i,j]
        cmd_j = cmd_prefix + ' ' + fname_config + ' ' + fname_pseudo_output + ' ' + fname_nmatrix_output + ' ' + str(i) + ' ' + str(j)
        jobname = job_prefix + str(i) + '-' + str(j)
        sh_file = jobname + '.sh'
        out_file = dir_outfiles + '/' + jobname + '.out'
        print(out_file)
        make_job(sh_file, out_file, jobname, cmd_j)
        #submit_job(sh_file)
        os.system('rm -f ' + sh_file)