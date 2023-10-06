import os.path
import sys
import numpy as np
import time
import datetime
import h5py
import configparser
import datetime
import matplotlib.pyplot as pl

sys.path.append('../')
from permittivity import southpole, exponential_profile
import util

sys.path.append('inversion')
from makeDepthScan import depth_scan

fname_config = 'config_aletsch_GA_pseudo.txt'
z0 = 7.5
n_ice = 1.78
n_surf = 1.3
A = n_ice
B = -(n_ice - n_surf)
C = -1/z0

zmin = 0
zmax = 30
dz = 0.05
zprof_sp = np.arange(zmin, zmax, dz)
nprof_sp0 = exponential_profile(zprof_sp, A, B, C)

def add_layer(n_prof, z_prof, n_layer, z_layer, dz_layer):
    z_min = z_layer - dz_layer/2
    z_max = z_layer + dz_layer/2
    ii_min = util.findNearest(z_prof, z_min)
    ii_max = util.findNearest(z_prof, z_max)
    n_prof_out = np.ones(len(z_prof))
    n_prof_out[:ii_min] = n_prof[:ii_min]
    n_prof_out[ii_min:ii_max] = n_layer
    n_prof_out[ii_max:] = n_prof[ii_max:]
    return n_prof_out

fig = pl.figure(figsize=(8,5),dpi=120)
ax = fig.add_subplot(111)
ax.plot(zprof_sp, nprof_sp0)

nprof_sp1 = add_layer(n_prof=nprof_sp0, z_prof=zprof_sp, n_layer=1.78, z_layer=6, dz_layer=0.2)
ax.plot(zprof_sp, nprof_sp1)
nprof_sp2 = add_layer(n_prof=nprof_sp1, z_prof=zprof_sp, n_layer=1.78, z_layer=8, dz_layer=0.2)
ax.plot(zprof_sp, nprof_sp2)

ax.grid()
fig.savefig('sp_with_layers.png')
pl.close(fig)

'''
depth_scan(fname_config=fname_config, n_profile=nprof_sp0, z_profile=zprof_sp, fname_out='exp_data.h5')
depth_scan(fname_config=fname_config, n_profile=nprof_sp1, z_profile=zprof_sp, fname_out='exp_data_1layer.h5')
depth_scan(fname_config=fname_config, n_profile=nprof_sp2, z_profile=zprof_sp, fname_out='exp_data_2layer.h5')
'''

#Test N_Layers
z_layer_list = np.arange(1.0, 14.0, 2.0)
fname0 = 'exp_data_'
for i in range(len(z_layer_list)):
    nprof_sp_i = add_layer(nprof_sp0, zprof_sp, 1.78, z_layer_list[i], .2)
    fname_i = fname0 + 'single_zlayer=' + str(z_layer_list[i]).zfill(3) + 'm.h5'
    depth_scan(fname_config=fname_config, n_profile=nprof_sp_i, z_profile=zprof_sp, fname_out=fname_i)
#TEST BRINGING LAYERS_TOGETHER
nprof_sp_list = []
nprof_sp_list.append(add_layer(nprof_sp0, zprof_sp, 1.78, z_layer_list[0], 0.2))
for i in range(1, len(z_layer_list)):
    nprof_sp_i = add_layer(nprof_sp_list[i-1], zprof_sp, 1.78, z_layer_list[i-1], .2)
    nprof_sp_list.append(nprof_sp_i)
    fname_i = fname0 + 'multi_zlayer=' + str(z_layer_list[i]).zfill(3) + 'm.h5'
    depth_scan(fname_config=fname_config, n_profile=nprof_sp_i, z_profile=zprof_sp, fname_out=fname_i)

#Test Seperation
z_seperation = [0.5, 1.0, 1.2, 1.5, 1.8, 2.0, 3.0]

for i in range(len(z_seperation)):
    z_layer2 = 6.0 + z_seperation[i]
    nprof_sp_i = add_layer(nprof_sp1, zprof_sp, 1.78, 6.0 + z_seperation[i], 0.2)
    fname_i = fname0 + '2layer_delta_z=' + str(z_seperation[i]) + 'm.h5'
    depth_scan(fname_config=fname_config, n_profile=nprof_sp_i, z_profile=zprof_sp, fname_out=fname_i)
