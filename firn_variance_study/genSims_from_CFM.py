import os.path
import sys
import numpy as np
import time
import datetime
import h5py
import configparser
from numpy import exp, log
import matplotlib.pyplot as pl


sys.path.append('../')
from paraPropPython import paraProp as ppp
import receiver as rx
from transmitter import tx_signal
from data import create_sim, create_rxList_from_file, create_tx_signal, create_transmitter_array
from data import create_hdf_bscan, bscan_rxList, create_hdf_FT
import util

fname_config = 'config_ICRC_summit.txt'
fname_nprofile = 'nProf_CFM.h5'
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
for i in range(nProfiles):
    nprof_in[i] = np.interp(z_prof_in, zprof_mat, nprof_mat[i]).real
''''
#print('is the n_prof nan?', np.any(np.isnan(nprof_mat[1])==True))
z_nan = []
for j in range(nDepths0):
    if np.isnan(nprof_mat[1,j]) == True:
        z_nan.append(zprof_mat[j])
print(z_nan)
#print('is the n_prof nan?', np.any(np.isnan(nprof_in[1])==True))
sys.exit()
'''

nSim = 5
tx_depths = create_transmitter_array(fname_config)
nTX = len(tx_depths)
rxList0 = create_rxList_from_file(fname_config)
tx_signal = create_tx_signal(fname_config)
nSamples = tx_signal.nSamples
nRX = len(rxList0)
'''
#fig = pl.figure(figsize=(8,5),dpi=120)
#ax = fig.add_subplot(111)
#for i in range(nSim):
#    ax.plot(zprof_mat, nprof_mat[i])
#ax.grid()
#pl.show()


freq1 = 0.2
sim = create_sim(fname_config)
sim.set_n(nVec=nprof_in[1],zVec=z_prof_in)

fig = pl.figure(figsize=(8,5),dpi=120)
ax = fig.add_subplot(111)
ax.plot(sim.get_z(), sim.get_n())
ax.grid()
pl.show()

sim.set_dipole_source_profile(centerFreq=freq1, depth=20)
fig = pl.figure(figsize=(8,5),dpi=120)
ax = fig.add_subplot(111)
ax.plot(sim.get_z(), sim.get_source_profile())
ax.grid()
pl.show()

sim.set_cw_source_signal(freq=freq1)
print(sim.A)
print(sim.get_source_profile())
sim.do_solver()

fig = pl.figure(figsize=(8,5),dpi=120)
ax = fig.add_subplot(111)
pmesh = ax.imshow(abs(np.transpose(sim.get_field())),  extent=[0, sim.x[-1], sim.z[-1], sim.z[0]], aspect='auto',cmap='viridis')
cbar = fig.colorbar(pmesh,ax=ax)
cbar.set_label('$P_{1}$ [dBm]')
ax.set_xlabel('Range x [m]')
ax.set_ylabel('Depth z [m]')
pl.show()
'''
for i in range(nSim):
    fname_out = 'sim_CFM_' + str(i).zfill(3) + '.h5'
    bscan_npy = np.zeros((nTX, nRX, nSamples), dtype='complex')

    for j in range(nTX):
        print('Sim', i, 'Tx_depths', tx_depths[j], 'm, id:', j)
        tstart = time.time()

        sourceDepth = tx_depths[j]
        sim = create_sim(fname_config)
        tx_signal = create_tx_signal(fname_config)
        tx_signal.get_gausspulse()
        rxList = create_rxList_from_file(fname_config)
        nprof_i = nprof_in[i]
        freq_centre = tx_signal.frequency
        nRX = len(rxList)

        sim.set_n(nVec=nprof_i, zVec=z_prof_in)
        sim.set_dipole_source_profile(centerFreq=freq_centre, depth=sourceDepth)
        sim.set_td_source_signal(sigVec=tx_signal.pulse, dt=tx_signal.dt)

        if j == 0:
            hdf_output = create_hdf_FT(fname=fname_out, sim=sim, tx_signal=tx_signal,
                                      tx_depths=tx_depths, rxList=rxList)

        sim.do_solver_smooth(rxList=rxList)
        tend = time.time()
        duration_s = (tend - tstart)
        duration = datetime.timedelta(seconds=duration_s)
        remainder_s = duration_s * (nTX - (i + 1))
        remainder = datetime.timedelta(seconds=remainder_s)

        completed = round(float(i + 1) / float(nTX) * 100, 2)
        print(completed, ' % completed, duration:', duration)
        print('remaining steps', nTX - (i + 1), '\nremaining time:', remainder, '\n')
        now = datetime.datetime.now()
        tstamp_now = now.timestamp()
        end_time = datetime.datetime.fromtimestamp(tstamp_now + remainder_s)
        print('completion at:', end_time)
        print('')
        for k in range(nRX):
            rx_k = rxList[k]
            sig_rx = rx_k.get_signal()
            bscan_npy[j,k] = sig_rx
    hdf_output.create_dataset('bscan_sig', data=bscan_npy)