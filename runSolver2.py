import numpy as np
from permittivity import *
import paraPropPython as ppp
import util
import time
import h5py
import sys
import datetime
import os

if len(sys.argv) != 3:
    print('error: should have format: python runSolver.py <output> <source-depth>')
    sys.exit()

output = str(sys.argv[1])
#print(output)
fname_h5 = output + '.h5'
output_h5 = h5py.File(fname_h5, 'r+')

fname_npy = output + '.npy'
output_npy = np.load(fname_npy, 'r+')

nProfile = np.load(output + '-nProf.npy', 'r')

#Check if input/output files exist
if (os.path.isfile(fname_h5) == True) and (os.path.isfile(fname_npy)) == True:
    pass
elif os.path.isfile(fname_h5) == False:
    print('No h5 file -> process aborted')
    sys.exit()
elif os.path.isfile(fname_npy) == False:
    print('No npy file -> process aborted')
    sys.exit()
else:
    print('no input or output files -> process aborted')
    sys.exit()

sourceDepth = float(sys.argv[2]) #source depth
iceDepth = output_h5.attrs["iceDepth"]
iceLength = output_h5.attrs["iceLength"]
airHeight0 = output_h5.attrs["airHeight"]
#output_hdf.attrs["airHeight"] = airHeight
dx = output_h5.attrs["dx"]
dz = output_h5.attrs["dz"]

tstart = time.time()
sim = ppp.paraProp(iceLength, iceDepth, dx, dz,airHeight=airHeight0, filterDepth=100, refDepth=sourceDepth)
tx_depths = output_h5["tx_depths"] #get data
rx_depths = output_h5["rx_depths"]

#Fill in blank
#load geometry
freqCentral = output_h5.attrs["freqCentral"]

mode = output_h5.attrs["mode"]
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
sim.set_td_source_signal(tx_pulse, dt)

#nTX = output_h5.attrs["nTX"]
RX_depths = np.array(output_h5.get("rx_depths"))
RX_ranges = np.array(output_h5.get("rx_ranges"))
nRX_depths = len(RX_depths)
nRX_ranges = len(RX_ranges)

rxArray = np.array(output_h5.get("rxArray"))
rxList = []
    ii = 0
    for i in range(nRX_ranges):
        for j in range(nRX_depths):
            rx_ij = ppp.receiver(rxArray[i,j,0], rxArray[i,j,1])
            rxList.append(rx_ij)
#print('mode', mode)
if mode == "1D":
    sim.do_solver(rxList)
elif mode == "2D":
    sim.do_solver2(rxList, freq_min = freqHP, freq_max = freqLP)
elif mode == "backwards_solver":
    sim.backwards_solver(rxList, freq_min = freqHP, freq_max = freqLP)

tx_depths = np.array(output_h5.get("tx_depths"))
ii_tx = util.findNearest(tx_depths, sourceDepth)
ii_freqHP = util.findNearest(freq_space, freqHP)
ii_freqLP = util.findNearest(freq_space, freqLP)

ii = 0
for i in range(len(RX_ranges)):
    for j in range(len(RX_depths)):
        rx_ij = rxList[ii]
        spectrum = rx_ij.spectrum
        output_npy[ii_tx, i, j, :] = spectrum
        ii += 1

tend = time.time()
duration = tend - tstart
solver_time = datetime.timedelta(seconds=duration)
completion_date = datetime.datetime.now()
date_str = completion_date.strftime("%d/%m/%Y %H:%M:%S")
print("simulation: " + sys.argv[2] + " src = " + sys.argv[2] + ", duration: " + str(solver_time) + " completed at: " + date_str)