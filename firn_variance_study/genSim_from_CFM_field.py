import os.path
import sys
import numpy as np
import time
import datetime
import h5py
import configparser
from numpy import exp, log
import matplotlib.pyplot as pl
from ku_scripting import *


sys.path.append('../')
from paraPropPython import paraProp as ppp
import receiver as rx
from transmitter import tx_signal
from data import create_sim, create_rxList_from_file, create_tx_signal, create_transmitter_array
from data import create_hdf_bscan, bscan_rxList, create_hdf_FT
import util

def get_dates(fname_in):
    with  h5py.File(fname_in, 'r') as input_hdf:
        data_matrix = np.array(input_hdf['density'])
    date_arr = data_matrix[1:,0]
    return date_arr


fname_config = 'config_ICRC_summit_km_ku.txt'
fname_nprofile = 'nProf_CFM_deep.h5'
fname_CFM = 'CFMresults.hdf5'

date_arr = get_dates(fname_CFM)

nprofile_hdf = h5py.File(fname_nprofile, 'r')
nprof_mat = np.array(nprofile_hdf.get('n_profile_matrix'))
zprof_mat = np.array(nprofile_hdf.get('z_profile'))

Depths0 = len(zprof_mat)
nProfiles = len(nprof_mat)

sim0 = create_sim(fname_config)
dz = sim0.dz
z_prof_in = np.arange(min(zprof_mat), max(zprof_mat), dz)
nDepths = len(z_prof_in)
nprof_in = np.ones((nProfiles, nDepths))

ii_cut = util.findNearest(zprof_mat, 100)
zprof_mat = zprof_mat[:ii_cut]
nprof_mat = nprof_mat[:,:ii_cut]

dir_sim = 'CFM_files_ku/fields_1D'
if os.path.isdir(dir_sim) == False:
    os.system('mkdir ' + dir_sim)
dir_sim_path = dir_sim + '/'


fname_CFM_list = 'CFM_sim_list.txt'
fout_list = open(fname_CFM_list,'w')

#start_year = int(date_arr[0])
start_year = 2015
end_year = int(date_arr[-1]) + 1
year_list = np.arange(start_year, end_year, 1)
year_id_list = []
print(year_list)
nYears = len(year_list)
fname_list = []
n_matrix_yr = np.ones((nYears, nDepths))
for i in range(nYears):
    jj = util.findNearest(date_arr, year_list[i])
    year_id_list.append(jj)
nProfiles_sim = nYears

z_depths = create_transmitter_array(fname_config)
freq_list = np.arange(0.1, 0.4, 0.1)
for i in year_id_list:
    for j in range(len(z_depths)):
        for k in range(len(freq_list)):
            print(date_arr[i], z_depths[j], freq_list[k])
            suffix = str(i).zfill(3) + 'z_' + str(round(z_depths[j],2)).zfill(3) + 'm_f_' + str(round(freq_list[k],2)).zfill(2)
            jobname = dir_sim_path + suffix
            fname_out0 ='sim_2CFM_' + suffix + '.h5'
            fname_out = dir_sim_path + fname_out0
            fname_sh_in = 'sim_2CFM_' + suffix + '.sh'
            fname_sh_out = dir_sim_path + 'sim_2CFM_' + suffix + '.out'
            z_tx = z_depths[j]
            freq = freq_list[k]

            cmd = 'python runSim_field.py ' + fname_config + ' ' + fname_nprofile + ' ' + str(i) + ' ' + str(z_tx) + ' ' + str(freq) + ' ' + fname_out
            make_job(fname_shell=fname_sh_in, fname_outfile=fname_sh_out, jobname=jobname, command=cmd)
            submit_job(fname_sh_in)
            line = fname_out0 + '\n'
            fout_list.write(line)
fout_list.close()
