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

from ascan_multi import *


fname_config = 'config_ICRC_summit_km_ku.txt'
fname_nprofile = 'nProf_CFM_deep.h5'
fname_CFM = 'CFMresults.hdf5'

date_arr = get_dates(fname_CFM)

dir_sim = 'CFM_files_ku'
dir_sim_path = dir_sim + '/'

start_year = 2011
end_year = int(date_arr[-1]) + 1
year_list = np.arange(start_year, end_year, 1)
year_id_list = []
print(year_list)
nYears = len(year_list)

nprofile_hdf = h5py.File(fname_nprofile, 'r')
nprof_mat = np.array(nprofile_hdf.get('n_profile_matrix'))
zprof_mat = np.array(nprofile_hdf.get('z_profile'))

Depths0 = len(zprof_mat)
nProfiles = len(nprof_mat)
sim0 = create_sim(fname_config)
dz = sim0.dz
z_prof_in = np.arange(min(zprof_mat), max(zprof_mat), dz)
tx_signal0 = create_tx_signal(fname_config)
freq_space = tx_signal0.freq_space

nDepths = len(z_prof_in)

n_matrix_yr = np.ones((nYears, nDepths))
for i in range(nYears):
    jj = util.findNearest(date_arr, year_list[i])
    year_id_list.append(jj)
nProfiles_sim = nYears

z_depths = create_transmitter_array(fname_config)

fname_list = dir_sim_path + 'CFM_sim_list.txt'
f_list = open(fname_list, 'w')

for i in year_id_list:
    fname_output_npy = dir_sim_path + 'sim_ascan_CFM_' + str(i).zfill(3) + '.npy'
    fname_output_h5 = dir_sim_path + 'sim_ascan_CFM_' + str(i).zfill(3) + '.h5'

    line = str(i) + '\t' + str(date_arr[i]) + '\t' + fname_output_npy
    line += fname_output_h5 + '\n'
    f_list.write(line)
    create_spectrum(fname_config=fname_config,
                    nprof_data=nprof_mat[i], zprof_data=zprof_mat,
                    fname_output_npy=fname_output_npy,
                    fname_output_h5=fname_output_h5)
    for j in range(len(freq_space)):
        for k in range(len(z_depths)):
            suffix =  str(i).zfill(3)+ 'z_' + str(round(z_depths[k],2)).zfill(3) + 'm_f_' + str(round(freq_space[j],2)).zfill(2)
            jobname = 'ascan_2CFM_' + suffix
            fname_sh_in = 'ascan_2CFM_' + suffix + '.sh'
            fname_sh_out = dir_sim_path + 'ascan_2CFM_' + suffix + '.out'
            cmd = 'python runSim_ascan.py ' + fname_config + ' ' + fname_output_npy + ' '
            cmd += fname_nprofile + ' ' + str(i) + ' ' + str(j) + ' ' + str(k)
            make_job(fname_shell=fname_sh_in, fname_outfile=fname_sh_out, jobname=jobname, command=cmd)
            submit_job(fname_sh_in)