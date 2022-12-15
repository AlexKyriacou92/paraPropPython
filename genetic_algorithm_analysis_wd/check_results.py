import sys
import numpy as np
import time
import datetime
import h5py
import matplotlib.pyplot as pl

sys.path.append('../')

import util
import paraPropPython as ppp
from receiver import receiver as rx
from transmitter import tx_signal
from data import create_sim, create_rxList_from_file, create_transmitter_array, create_hdf_FT
from data import create_tx_signal, bscan

fname_results = sys.argv[1]
hdf_results = h5py.File(fname_results, 'r')

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

for i in range(nGenerations):
    S_best[i] = max(S_results[i])
    best_individuals.append(np.argmax(S_results[i]))

ii_best_gen = np.argmax(S_best)
jj_best_ind = best_individuals[ii_best_gen]
n_profile_best = n_profile_matrix[ii_best_gen, jj_best_ind]
fig = pl.figure(figsize=(8,5),dpi=120)
pl.plot(gens, S_best,c='b')
pl.xlabel('Generation')
pl.ylabel(r'Best score $S_{max}$')
pl.grid()
pl.show()

fig = pl.figure(figsize=(4,10),dpi=120)
pl.title(r'Generation: '+ str(ii_best_gen) + r', Individual: ' + str(jj_best_ind) + r', S = ' + str(round(S_best[ii_best_gen]/1e6,2)) + r' $ \times 10^{6}$')
pl.plot(n_profile_best, z_profile, '-o', c='b')
pl.grid()
pl.ylim(16,-1)
pl.ylabel(r'Depth z [m]')
pl.xlabel(r'Refractive Index Profile $n(z)$')
pl.show()

rho_profile_best = (n_profile_best - 1)/0.835 * 1e3
fig = pl.figure(figsize=(4,10),dpi=120)
pl.title(r'Generation: '+ str(ii_best_gen) + r', Individual: ' + str(jj_best_ind) + r', S = ' + str(round(S_best[ii_best_gen]/1e6,2)) + r' $ \times 10^{6}$')
pl.plot(rho_profile_best, z_profile, '-o', c='b')
pl.grid()
pl.ylim(16,-1)
pl.ylabel(r'Depth z [m]')
pl.xlabel(r'Density Profile $\rho(z)$ [$\mathrm{kg/m^{3}}$]')
pl.show()