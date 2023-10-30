import os.path
import sys
import numpy as np
import time
import datetime
import h5py
import configparser
from numpy import exp, log
import matplotlib.pyplot as pl
from makeDepthScan import depth_scan_impulse_smooth
from pleiades_scripting import *

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
nDepths0 = len(zprof_mat)
nProfiles = len(nprof_mat)

sim0 = create_sim(fname_config)
dz = sim0.dz
z_prof_in = np.arange(min(zprof_mat), max(zprof_mat), dz)
nDepths = len(z_prof_in)
nprof_in = np.ones((nProfiles, nDepths))

ii_cut = util.findNearest(zprof_mat, 100)
zprof_mat = zprof_mat[:ii_cut]
nprof_mat = nprof_mat[:,:ii_cut]

dir_sim = 'CFM_files'
if os.path.isdir(dir_sim) == False:
    os.system('mkdir ' + dir_sim)
dir_sim_path = dir_sim + '/'

fname_CFM_list = 'CFM_sim_list.txt'
fout_list = open(fname_CFM_list,'w')
'''
if len(sys.argv) > 1:
    nProfiles_sim = int(sys.argv[1])
else:
    nProfiles_sim = nProfiles
year_list = np.arange(start_year, end_year, 1)
'''
#start_year = int(date_arr[0])
start_year = 2011
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
for i in year_id_list:
    print(i)
    jobname = dir_sim_path + 'sim_2CFM_' + str(i).zfill(3)
    fname_out0 ='sim_2CFM_' + str(i).zfill(3) + '.h5'
    fname_out = dir_sim_path + fname_out0
    fname_sh_in = 'sim_2CFM_' + str(i).zfill(3) + '.sh'
    fname_sh_out = dir_sim_path + 'sim_2CFM_' + str(i).zfill(3) + '.out'

    cmd = 'python runSim_impulse.py ' + fname_config + ' ' + fname_nprofile + ' ' + str(i) + ' ' + fname_out
    make_job(fname_shell=fname_sh_in, fname_outfile=fname_sh_out, jobname=jobname, command=cmd)
    submit_job(fname_sh_in)
    line = fname_out0 + '\n'
    fout_list.write(line)
fout_list.close()
