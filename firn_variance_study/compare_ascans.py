import numpy as np
import matplotlib.pyplot as pl
import sys

sys.path.append('../')
from data import bscan, bscan_rxList

config_args = 4
min_args = config_args + 1
nArgs = len(sys.argv)
nFile = nArgs - config_args

if nArgs < min_args:
    print('input error, script', sys.argv[0], ' requires at least ', min_args, 'arguments')
    print('form: python ', sys.argv[0], ' <zTx> <xRx> <zRx> fname_1 ...')
    sys.exit()

zTx = float(sys.argv[1])
xRx = float(sys.argv[2])
zRx = float(sys.argv[3])

fname_list = []
legend_list = []

for i in range(config_args, nArgs):
    fname = sys.argv[i]
    legend_label = fname[:-3]
    fname_list.append(sys.argv[i])
    legend_list.append(legend_label)

fname_h5 = fname_list[0]
sim_bscan = bscan_rxList()
sim_bscan.load_sim(fname_h5)
ascan = sim_bscan.get_ascan_from_depth(z_tx=zTx, x_rx=xRx, z_rx=zRx)
zTx2, xRx2, zRx2 = sim_bscan.get_actual_position(z_tx=zTx, x_rx=xRx, z_rx=zRx)

ii_max = np.argmax(abs(ascan))
t_peak_max = sim_bscan.tspace[ii_max]
tmin = t_peak_max - 100
tmax = t_peak_max + 300

fig = pl.figure(figsize=(10,6),dpi=150)
ax = fig.add_subplot(111)
ax.set_title('R = ' + str(round(xRx2,2)) + ' m, z_tx = ' + str(round(zTx2,2)) + ' m, z_rx = ' + str(round(zRx2,2)))
for i in range(nFile):
    fname_h5 = fname_list[i]
    sim_bscan = bscan_rxList()
    sim_bscan.load_sim(fname_h5)
    ascan = sim_bscan.get_ascan_from_depth(z_tx=zTx, x_rx=xRx, z_rx=zRx)
    tspace = sim_bscan.tspace
    ax.plot(tspace, ascan.real, label=legend_list[i])
ax.grid()
ax.set_xlabel('Time [ns]')
ax.set_ylabel('Amplitude [V]')
ax.legend()
ax.set_xlim(tmin,tmax)

fig.savefig('compare_plots_pulse.png')
pl.close(fig)

#==============================================================================================

fig = pl.figure(figsize=(10,6),dpi=150)
ax = fig.add_subplot(111)
ax.set_title('R = ' + str(round(xRx2,2)) + ' m, z_tx = ' + str(round(zRx2,2)) + ' m, z_rx = ' + str(round(zRx2,2)))
for i in range(nFile):
    print(i)
    fname_h5 = fname_list[i]
    sim_bscan = bscan_rxList()
    sim_bscan.load_sim(fname_h5)
    ascan = sim_bscan.get_ascan_from_depth(z_tx=zTx, x_rx=xRx, z_rx=zRx)
    tspace = sim_bscan.tspace
    ax.plot(tspace, abs(ascan), label=legend_list[i])
ax.grid()
ax.set_xlabel('Time [ns]')
ax.set_ylabel('Amplitude [u]')
ax.legend()
ax.set_xlim(tmin,tmax)

fig.savefig('compare_plots_abs.png')
pl.close(fig)

#==============================================================================================

fig = pl.figure(figsize=(10,6),dpi=150)
ax = fig.add_subplot(111)
ax.set_title('R = ' + str(round(xRx2,2)) + ' m, z_tx = ' + str(round(zRx2,2)) + ' m, z_rx = ' + str(round(zRx2,2)))
for i in range(nFile):
    print(i)
    fname_h5 = fname_list[i]
    sim_bscan = bscan_rxList()
    sim_bscan.load_sim(fname_h5)
    ascan = sim_bscan.get_ascan_from_depth(z_tx=zTx, x_rx=xRx, z_rx=zRx)
    tspace = sim_bscan.tspace
    ax.plot(tspace, 20*np.log10(abs(ascan)), label=legend_list[i])
ax.grid()
ax.set_xlabel('Time [ns]')
ax.set_ylabel('Power [dBu]')
ax.legend()
ax.set_xlim(tmin,tmax)

fig.savefig('compare_plots_dB.png')
pl.close(fig)