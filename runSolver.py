import numpy as np
from permittivity import *
from geometry import triangle, circle
import paraPropPython as ppp
from paraPropPython import receiver as rx
import util
from backwardsSolver import backwards_solver
from matplotlib import pyplot as pl

from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
import permittivity as epsilon
import time

import h5py
import sys
import datetime
import os

#Info on this file:
"""

Stages of the simulations
1. 
Initiate Simulation
python genSim.py <input.txt> <output>

-> creates output.h5 and output.npy
-> if you're overriding a previous simulation (h5-file) it will ask for you for permission to over-write
-> if yes, then it will proceed and create a npy file
-> if no, then no npy file will be generated -> which will disable the next stage

2.
Submit the solver jobs to queue -> write to a npy memmap
for tx in range(nTransmitters):
    for freq in range(freq_min, freq_max):
        python runSolver.py <freq> <source-depth> -node <output>

3. 
Check if simulation was successful (??)
Writes the memmap to the h5 file
python saveSim.py <output>
deletes npy file

runSolver.py solves PE for a single frequency and source depth

usage:
python mpiSim.py <freq> <source depth> <output> 
<h5 file> : holds all relevant data 

"""

if len(sys.argv) != 4:
    print('error: should have format: python runSolver.py <output> <freq> <source-depth>')
    sys.exit()

output = str(sys.argv[1])
#print(output)
fname_h5 = output + '.h5'
output_h5 = h5py.File(fname_h5, 'r')
nProfile = np.load(output + '-nProf.npy', 'r')


freq = float(sys.argv[2]) #Frequency of Simulation
sourceDepth = float(sys.argv[3]) #source depth

iceDepth = output_h5.attrs["iceDepth"]
iceLength = output_h5.attrs["iceLength"]
airHeight0 = output_h5.attrs["airHeight"]
#output_hdf.attrs["airHeight"] = airHeight
dx = output_h5.attrs["dx"]
dz = output_h5.attrs["dz"]

tstart = time.time()
sim = ppp.paraProp(iceLength, iceDepth, dx, dz,airHeight=airHeight0, filterDepth=100, refDepth=sourceDepth)

x = sim.get_x()
z = sim.get_z()
zFull = sim.zFull
tx_depths = output_h5["tx_depths"] #get data
rx_depths = output_h5["rx_depths"]

#Fill in blank
#load geometry
freqCentral = output_h5.attrs["freqCentral"]

mode = output_h5.attrs["mode"]
#print('mode:',mode)
#print(nProfile.shape)
if mode == "1D":
    sim.set_n(method='vector', nVec=nProfile)
else:
    sim.set_n2(method='matrix',nMat=nProfile)

#Set profile
sim.set_dipole_source_profile(freqCentral, sourceDepth)

freqLP = output_h5.attrs["freqLP"]
freqHP = output_h5.attrs["freqHP"]
nSamples = output_h5.attrs["nSamples"]
dt = output_h5.attrs["dt"]

freq_space = np.fft.fftfreq(nSamples, dt)
freq_list = np.arange(freqLP, freqHP, nSamples)

#tx pulse
tx_pulse = np.array(output_h5.get("signalPulse"))
tx_spectrum = np.array(output_h5.get("signalSpectrum"))

#amplitude
ii_freq = util.findNearest(freq_space, freq)
amplitude = tx_spectrum[ii_freq]

#Set CW
sim.set_cw_source_signal(freq, amplitude)
rxList = np.array(output_h5.get("rxList"))
#print('mode', mode)
if mode == "1D":
    sim.do_solver()
elif mode == "2D":
    sim.do_solver2()
elif mode == "backwards_solver":
    sim.backwards_solver_2way()

### plot absolute value of field for whole simulation space ###
fig = pl.figure()
ax = fig.add_subplot(111)
#pl.imshow(10*np.log10(np.transpose(abs(sim.get_field()))), aspect='auto', cmap='hot',  vmin=-50, vmax=-20, extent=(x[0], x[-1], zFull[-1], zFull[0])) #dB Amplitude
#pl.imshow(20*np.log10(np.transpose(abs(sim.get_field()))), aspect='auto', cmap='hot',  vmin=-100, vmax=-40, extent=(x[0], x[-1], zFull[-1], zFull[0])) #dB Power
pmesh = pl.imshow(np.transpose(abs(sim.get_field())), aspect='auto', cmap='hot',  vmin=1e-5, vmax=1e-2, extent=(x[0], x[-1], zFull[-1], zFull[0]))
cbar = pl.colorbar(pmesh)
cbar.set_label()
pl.title("Absolute Field, " + str(int(freq*1000))+" MHz")
pl.ylim(z[-1],z[0])
pl.xlabel("x (m)")
pl.ylabel("z (m)")
pl.show()

"""
tx_depths = np.array(output_h5.get("tx_depths"))
ii_tx = util.findNearest(tx_depths, sourceDepth)

#nTX = output_h5.attrs["nTX"]
RX_depths = np.array(output_h5.get("rx_depths"))
RX_ranges = np.array(output_h5.get("rx_ranges"))

ii = 0
for i in range(len(RX_ranges)):
    for j in range(len(RX_depths)):
        x_rx = RX_ranges[i]
        z_rx = RX_depths[j]
        field_amp = sim.get_field(x0=x_rx, z0=z_rx)
        #print(x_rx, z_rx, field_amp)
        output_npy[ii_tx, i, j, ii_freq] = field_amp
        #amp_ij = rxList[ii]
        #output_npy[ii_tx, i, j, ii_freq] = rxList[ii].get_amplitude()
        #ii += 1

tend = time.time()
duration = tend - tstart
solver_time = datetime.timedelta(seconds=duration)
completion_date = datetime.datetime.now()
date_str = completion_date.strftime("%d/%m/%Y %H:%M:%S")
print("simulation: " + sys.argv[2] + " " + sys.argv[2] + " " + sys.argv[3] + ", duration: " + str(solver_time) + " completed at: " + date_str)
"""