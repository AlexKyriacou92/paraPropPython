import numpy as np
import matplotlib.pyplot as pl
import sys

sys.path.append('../')
from data import bscan

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

fig = pl.figure(figsize=(10,6),dpi=150)
ax = fig.add_subplot(111)
ax.set_title('R = ' + str(round(xRx,2)) + ' m, z_tx = ' + str(round(zTx,2)) + ' m, z_rx = ' + str(round(zRx,2)))
for i in range(nFile):
    fname_h5 = fname_list[i]
    sim_bscan = bscan()
    sim_bscan.load_sim(fname_h5)
    ascan = sim_bscan.get_ascan(zTx=zTx, xRx=xRx, zRx=zRx)
    tspace = sim_bscan.tspace
    ax.plot(tspace, ascan.real, label=legend_list[i])
ax.grid()
ax.set_xlabel('Time [ns]')
ax.set_ylabel('Amplitude [V]')
ax.legend()
ax.set_xlim(0,300)

fig.savefig('plots/compare_plots_pulse.png')
pl.close(fig)

#==============================================================================================

fig = pl.figure(figsize=(10,6),dpi=150)
ax = fig.add_subplot(111)
ax.set_title('R = ' + str(round(xRx,2)) + ' m, z_tx = ' + str(round(zRx,2)) + ' m, z_rx = ' + str(round(zRx,2)))
for i in range(nFile):
    print(i)
    fname_h5 = fname_list[i]
    sim_bscan = bscan()
    sim_bscan.load_sim(fname_h5)
    ascan = sim_bscan.get_ascan(zTx=zTx, xRx=xRx, zRx=zRx)
    tspace = sim_bscan.tspace
    ax.plot(tspace, abs(ascan), label=legend_list[i])
ax.grid()
ax.set_xlabel('Time [ns]')
ax.set_ylabel('Amplitude [u]')
ax.legend()
ax.set_xlim(0,300)

fig.savefig('plots/compare_plots_abs.png')
pl.close(fig)

#==============================================================================================

fig = pl.figure(figsize=(10,6),dpi=150)
ax = fig.add_subplot(111)
ax.set_title('R = ' + str(round(xRx,2)) + ' m, z_tx = ' + str(round(zRx,2)) + ' m, z_rx = ' + str(round(zRx,2)))
for i in range(nFile):
    print(i)
    fname_h5 = fname_list[i]
    sim_bscan = bscan()
    sim_bscan.load_sim(fname_h5)
    ascan = sim_bscan.get_ascan(zTx=zTx, xRx=xRx, zRx=zRx)
    tspace = sim_bscan.tspace
    ax.plot(tspace, 20*np.log10(abs(ascan)), label=legend_list[i])
ax.grid()
ax.set_xlabel('Time [ns]')
ax.set_ylabel('Power [dBu]')
ax.legend()
ax.set_xlim(0,300)

fig.savefig('plots/compare_plots_dB.png')
pl.close(fig)