import os.path
import sys
import numpy as np
import time
import datetime
import h5py
import configparser

from makeDepthScan import depth_scan_from_hdf, depth_scan_from_hdf_IR

sys.path.append('../')

import util
import paraPropPython as ppp
from receiver import receiver as rx
from transmitter import tx_signal
from data import create_sim, create_rxList_from_file, create_transmitter_array, create_hdf_FT
from data import create_tx_signal, bscan, bscan_rxList, create_hdf_bscan

fname_config = sys.argv[1]
fname_n_matrix = sys.argv[2]
path2sim = sys.argv[3]

nmatrix_hdf = h5py.File(fname_n_matrix, 'r')
n_profile_matrix = np.array(nmatrix_hdf['n_profile_matrix'])
S_arr = np.array(nmatrix_hdf['S_arr'])
nGenerations = len(S_arr)
nIndividuals = len(S_arr[0])
genes_arr = np.array(nmatrix_hdf['genes_matrix'])
z_genes = np.array(nmatrix_hdf['z_genes'])
nGenes = len(z_genes)
z_profile = abs(np.array(nmatrix_hdf['z_profile']))[1:-1]
nmatrix_hdf.close()

nGenerations_complete = 0
S_max_list = []
i_gen_list = []
j_max_list = []
for i in range(nGenerations):
    if np.all(S_arr[i] == 0) == False:
        S_max_list.append(max(S_arr[i]))
        j_max = np.argmax(S_arr[i])
        i_gen_list.append(i)
        j_max_list.append(j_max)
        nGenerations_complete += 1

i_max = np.argmax(S_max_list)

print(i_max)
ii_max = i_gen_list[i_max]
jj_max = j_max_list[i_max]
S_max = S_max_list[i_max]

if os.path.isdir(path2sim) == False:
    os.system('mkdir ' + path2sim)
fname_out = path2sim + '/' + 'sim_gen' + str(ii_max) + '_ind' + str(jj_max) + '_bscan.h5'

depth_scan_from_hdf_IR(fname_config=fname_config, fname_n_matrix=fname_n_matrix,
                    ii_generation=ii_max, jj_select=jj_max,
                    fname_pseudo = None, fname_out=fname_out)