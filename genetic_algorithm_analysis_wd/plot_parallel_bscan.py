import numpy as np
import sys
sys.path.append('../')

import paraPropPython as ppp
from receiver import receiver
from data import bscan, bscan_rxList

import scipy.signal as signal
import util
import h5py
import matplotlib.pyplot as pl
import peakutils as pku
import sys

if len(sys.argv) == 3:
    fname = sys.argv[1]
    fname_FT = sys.argv[2]
else:
    print('error: must enter python plot_parallel_bscan.py fname')
    sys.exit()
vmin0 = -100
vmax0 = 10

fname_h5 = fname
sim_bscan = bscan_rxList()
sim_bscan.load_sim(fname_h5)
tspace = sim_bscan.tx_signal.tspace
tx_depths = sim_bscan.tx_depths
nSamples = sim_bscan.nSamples
nDepths = len(tx_depths)
rxList = sim_bscan.rxList
print(rxList[0].x)

rx_depths = []
for i in range(len(rxList)):
    rx_depths.append(rxList[i].z)

bscan_sig = sim_bscan.bscan_sig

bscan_plot = np.zeros((nDepths, nSamples), dtype='complex')

hdf_FT = h5py.File(fname_FT,'r')
fftArray = np.array(hdf_FT.get('fftArray'))
tx_depths_FT = np.array(hdf_FT.get('txDepths'))
rx_depths_FT = np.array(hdf_FT.get('rxDepths'))
nDepths_FT = len(tx_depths_FT)
bscan_FT = np.zeros((nDepths, nSamples), dtype='complex')
hdf_FT.close()
for i in range(nDepths):
    jj_rx = util.findNearest(rx_depths, tx_depths[i])
    bscan_plot[i] = bscan_sig[i,jj_rx]

    ii_tx_FT = util.findNearest(tx_depths_FT, tx_depths[i])
    ii_rx_FT = util.findNearest(rx_depths_FT, tx_depths[i])
    bscan_FT[i] = fftArray[ii_tx_FT]

fig = pl.figure(figsize=(8,6),dpi=200)
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)
ax1.set_title('Paralell Bscan [dBu]')
pmesh = ax1.imshow(10*np.log10((abs(bscan_plot)**2)), aspect='auto', extent=[0, tspace[-1], tx_depths[-1], tx_depths[0]], vmin=vmin0, vmax=vmax0)
#cbar = pl.colorbar(pmesh)
#cbar.set_label('Power P [dBu]')
ax1.set_ylabel(r'Depth $Z_{tx}$ [m]')
ax1.set_xlabel(r'$t$ [ns]')
ax1.set_ylim(15,1)


ax2.set_title('Paralell Bscan [dBu]')
pmesh = ax2.imshow(10*np.log10((abs(bscan_FT)**2)), aspect='auto', extent=[0, tspace[-1], tx_depths[-1], tx_depths[0]])
#cbar = pl.colorbar(pmesh)
#cbar.set_label('Power P [dBu]')
ax2.set_ylabel(r'Depth $Z_{tx}$ [m]')
ax2.set_xlabel(r'$t$ [ns]')
ax2.set_ylim(15,1)

pl.savefig(fname + '-bscan.png')
pl.show()
