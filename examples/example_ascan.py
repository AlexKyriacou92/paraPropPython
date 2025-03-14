import datetime
import numpy as np
import sys
import time
import os
from os.path import join

sys.path.append('../')
import paraPropPython as ppp
from receiver import receiver
from data import bscan, ascan
from permittivity import southpole

from transmitter import tx_signal

iceDepth = 100.
iceLength = 100.

dx = 1.0
dz = 0.05
airHeight = 10.

freq = 0.2
fmin = 0.05
fmax = 0.3
band = fmax-fmin
t_centre = 10.
dt = 1
t_max = 3 * 1.8 * iceLength/0.3

path2data = 'ascan_dir'
if os.path.isdir(path2data) == False:
    os.system('mkdir ' + path2data)

d_rx = 10
rx_ranges = np.arange(d_rx, iceLength, d_rx)
rx_depths = np.arange(d_rx, iceDepth, d_rx)


z_tx = [50.]
nTx = len(z_tx)

for j in range(nTx):
    sourceDepth = z_tx[j]
    print('z_tx = ', sourceDepth, 'm')
    fname_out = 'ascan_ztx_' + str(sourceDepth) + '_Xice_' + str(iceLength) + '_Zice_' + str(iceDepth) + '.h5'
    fname_out = join(path2data, fname_out)
    tx_signal0 = tx_signal(frequency=freq, bandwidth=band, t_centre=t_centre, dt=dt, tmax=t_max, amplitude=1)
    tx_signal0.set_impulse()
    tx_signal0.apply_bandpass(fmin=fmin, fmax=fmax)
    tx_pulse = tx_signal0.pulse.real
    tspace = tx_signal0.tspace

    rxList = []
    for x in rx_ranges:
        for z in rx_depths:
            rx_ij = receiver(x=x, z=z)
            rxList.append(rx_ij)

    sim = ppp.paraProp(iceLength=iceLength, iceDepth=iceDepth, dx=dx, dz=dz, airHeight=airHeight)
    sim.set_n(nFunc=southpole)
    sim.set_dipole_source_profile(centerFreq=freq, depth=sourceDepth)
    sim.set_td_source_signal(sigVec=tx_pulse, dt=dt)
    tstart = time.time()
    sim.do_solver(rxList)
    tend = time.time()

    duration = tend - tstart
    print('Solution time:')
    print(datetime.timedelta(seconds=duration))

    ascan_1 = ascan()
    ascan_1.save_sim_to_hdf(sim=sim,
                            tx_signal_in=tx_signal0,
                            rxList=rxList,
                            sourceDepth=sourceDepth,
                            fname_hdf=fname_out)