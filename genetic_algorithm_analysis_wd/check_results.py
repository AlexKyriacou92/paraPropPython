import sys
import numpy as np
import time
import datetime
import h5py
import matplotlib.pyplot as pl
from scipy.interpolate import interp1d

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
n_profile_ref = np.array(hdf_results.get('reference_data'))

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
profile_data = np.genfromtxt(fname_txt)
nprof_data = profile_data[:,1]
zprof_data = profile_data[:,0]
'''
nprof_data = n_profile_ref
zprof_data = z_profile
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
pl.plot(n_profile_best, z_profile, c='b',label='Best Result')
pl.plot(nprof_data, zprof_data,c='k',label='truth')
pl.grid()
pl.ylim(16,-1)
pl.ylabel(r'Depth z [m]')
pl.xlabel(r'Refractive Index Profile $n(z)$')
pl.legend()
pl.show()

f_interp_data = interp1d(zprof_data, nprof_data)
z_space = np.linspace(min(z_profile), max(z_profile), len(z_profile))
ii_min = util.findNearest(zprof_data, min(z_profile))
ii_max = util.findNearest(zprof_data, max(z_profile))
n_space_interp = np.ones(len(z_profile))
n_space_interp[0] = nprof_data[ii_min]
n_space_interp[-1] = nprof_data[ii_max]
n_space_interp[1:-1] = f_interp_data(z_profile[1:-1])

n_residuals = n_profile_best - n_space_interp

fig = pl.figure(figsize=(4,10),dpi=120)
pl.title(r'Generation: '+ str(ii_best_gen) + r', Individual: ' + str(jj_best_ind) + r', S = ' + str(round(S_best[ii_best_gen]/1e6,2)) + r' $ \times 10^{6}$')
pl.plot(n_residuals, z_profile, c='b',label='Residuals')
pl.axvline(0,c='r')
pl.fill_betweenx(z_profile, -0.05, +0.05,color='r',alpha=0.5,label='Boundary +/- 5%')
pl.grid()
pl.ylim(16,-1)
pl.ylabel(r'Depth z [m]')
pl.xlabel(r'Refractive Index Profile (residuals) $\Delta n(z)$')
pl.legend()
pl.show()

'''
rho_profile_best = (n_profile_best - 1)/0.835 * 1e3
fig = pl.figure(figsize=(4,10),dpi=120)
pl.title(r'Generation: '+ str(ii_best_gen) + r', Individual: ' + str(jj_best_ind) + r', S = ' + str(round(S_best[ii_best_gen]/1e6,2)) + r' $ \times 10^{6}$')
pl.plot(rho_profile_best, z_profile, '-o', c='b')
pl.grid()
pl.ylim(16,-1)
pl.ylabel(r'Depth z [m]')
pl.xlabel(r'Density Profile $\rho(z)$ [$\mathrm{kg/m^{3}}$]')
pl.show()
'''