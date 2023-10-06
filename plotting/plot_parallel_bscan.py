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
    R = float(sys.argv[2])
else:
    print('error: must enter python plot_parallel_bscan.py fname R')
    sys.exit()
vmin0 = -120
vmax0 = 0

fname_h5 = fname
sim_bscan = bscan_rxList()
sim_bscan.load_sim(fname_h5)
tspace = sim_bscan.tx_signal.tspace
#rx_ranges = sim_bscan.rx_ranges
tx_depths = sim_bscan.tx_depths
#rxList = sim_bscan.rxList

bscan_plot = sim_bscan.bscan_parallel(R)
#ii_rx_x = util.findNearest(rx_ranges, R)

fig = pl.figure(figsize=(8,6),dpi=200)
ax = fig.add_subplot(111)
ax.set_title('Paralell Bscan, dBu')
pmesh = ax.imshow(10*np.log10((abs(bscan_plot)**2)), aspect='auto', extent=[0, tspace[-1], tx_depths[-1], tx_depths[0]], vmin=vmin0, vmax=vmax0)
cbar = pl.colorbar(pmesh)
cbar.set_label('Power P [dBu]')
ax.set_ylabel(r'Depth $Z_{tx}$ [m]')
ax.set_xlabel(r'$t$ [ns]')
ax.set_ylim(15,1)
pl.savefig(fname + '-bscan.png')
pl.show()