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
import multiprocessing as mpi

if len(sys.argv) != 2:
    print('error: should have format: python runSolver.py <output>')
    sys.exit()

output = str(sys.argv[1])
print(output)
fname_h5 = output + '.h5'
output_h5 = h5py.File(fname_h5, 'r+')

fname_npy = output + '.npy'
output_npy = np.load(fname_npy, 'r+')

nProfile = np.load(output +  '-nProf.npy', 'r')

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


def get_args(output_h5, sourceDepth):
    args = []

    iceDepth = output_h5.attrs["iceDepth"]
    args.append(iceDepth)
    iceLength = output_h5.attrs["iceLength"]
    args.append(iceLength)

    airHeight0 = output_h5.attrs["airHeight"]
    args.append(airHeight0)
    # output_hdf.attrs["airHeight"] = airHeight
    dx = output_h5.attrs["dx"]
    args.append(dx)
    dz = output_h5.attrs["dz"]
    args.append(dz)

    tx_depths = np.array(output_h5.get("tx_depths"))  # get data
    args.append(tx_depths)
    nTX = len(tx_depths)

    rx_depths = np.array(output_h5.get("rx_depths"))
    args.append(rx_depths)

    nRX_depths = len(rx_depths)

    rx_ranges = np.array(output_h5.get("rx_ranges"))
    args.append(rx_ranges)
    nRX_ranges = len(rx_ranges)

    # Fill in blank
    # load geometry
    freqCentral = output_h5.attrs["freqCentral"]
    args.append(freqCentral)

    mode = output_h5.attrs["mode"]
    args.append(mode)
    print('mode:', mode)
    print(nProfile.shape)

    # tx pulse
    tx_pulse = np.array(output_h5.get("signalPulse"))
    args.append(tx_pulse)
    # tx_spectrum = np.array(output_h5.get("signalSpectrum"))

    freqLP = output_h5.attrs["freqLP"]
    args.append(freqLP)
    freqHP = output_h5.attrs["freqHP"]
    args.append(freqHP)
    nSamples = output_h5.attrs["nSamples"]
    args.append(nSamples)
    dt = output_h5.attrs["dt"]
    args.append(dt)

    rxArray = np.array(output_h5.get("rxArray"))
    print(rxArray)
    args.append(rxArray)
    rxList = []
    ii = 0
    args.append(sourceDepth)
    # print('rxList = ', rxList)
    return args

def solver(args):
    iceDepth = args[0]
    iceLength = args[1]
    airHeight0 = args[2]
    dx = args[3]
    dz = args[4]
    tx_depths = args[5]
    nTX = len(tx_depths)

    rx_depths = args[6]
    nRX_depths = len(rx_depths)

    rx_ranges = args[7]
    nRX_ranges = len(rx_ranges)

    freqCentral = args[8]
    mode = args[9]
    tx_pulse = args[10]
    freqLP = args[11]
    freqHP = args[12]

    nSamples = args[13]
    dt = args[14]
    rxArray = args[15]
    sourceDepth = args[16]
    """
    #output_h5, sourceDepth
    output_h5 = args[0]
    sourceDepth = args[1]

    iceDepth = output_h5.attrs["iceDepth"]
    iceLength = output_h5.attrs["iceLength"]
    airHeight0 = output_h5.attrs["airHeight"]
    # output_hdf.attrs["airHeight"] = airHeight
    dx = output_h5.attrs["dx"]
    dz = output_h5.attrs["dz"]

    tx_depths = np.array(output_h5.get("tx_depths"))  # get data
    nTX = len(tx_depths)

    rx_depths = np.array(output_h5.get("rx_depths"))
    nRX_depths = len(rx_depths)

    rx_ranges = np.array(output_h5.get("rx_ranges"))
    nRX_ranges = len(rx_ranges)

    # Fill in blank
    # load geometry
    freqCentral = output_h5.attrs["freqCentral"]

    mode = output_h5.attrs["mode"]
    print('mode:', mode)
    print(nProfile.shape)

    # tx pulse
    tx_pulse = np.array(output_h5.get("signalPulse"))
    #tx_spectrum = np.array(output_h5.get("signalSpectrum"))

    freqLP = output_h5.attrs["freqLP"]
    freqHP = output_h5.attrs["freqHP"]
    nSamples = output_h5.attrs["nSamples"]
    dt = output_h5.attrs["dt"]

    rxArray = np.array(output_h5.get("rxArray"))
    """
    print(rxArray)
    rxList = []
    ii = 0
    for i in range(nRX_ranges):
        for j in range(nRX_depths):
            rx_ij = ppp.receiver(rxArray[i,j,0], rxArray[i,j,1])
            rxList.append(rx_ij)
    #print('rxList = ', rxList)
    # ===========================

    sim = ppp.paraProp(iceLength, iceDepth, dx, dz, airHeight=airHeight0, filterDepth=100, refDepth=sourceDepth)
    # Set profile


    if mode == "1D":
        sim.set_n(method='vector', nVec=nProfile)
    else:
        sim.set_n2(method='matrix', nMat=nProfile)
    sim.set_td_source_signal(tx_pulse, dt)
    if mode == "1D":
        sim.do_solver(rxList)
    elif mode == "2D":
        sim.do_solver2(rxList, freq_min = freqHP, freq_max = freqLP)
    elif mode == "backwards_solver":
        sim.backwards_solver(rxList, freq_min = freqHP, freq_max = freqLP)

    sim.set_dipole_source_profile(freqCentral, sourceDepth)
    ii_source = util.findNearest(tx_depths, sourceDepth)

    freq_space = np.fft.fftfreq(nSamples, dt)
    ii_rxList = 0
    ii_freqHP = util.findNearest(freq_space, freqHP)
    ii_freqLP = util.findNearest(freq_space, freqLP)

    for i in range(nRX_ranges):
        for j in range(nRX_depths):
            rx = rxList[ii_rxList]
            spectrum = rx.spectrum
            output_npy[ii_source, i, j, ii_freqHP:ii_freqLP] = spectrum[ii_freqHP:ii_freqLP]
            ii_rxList += 1

if __name__ == '__main__':
    tx_depths = np.array(output_h5.get("tx_depths"))
    nTX = output_h5.attrs["nTransmitters"]
    arg_list = []
    for i in range(nTX):
        args = get_args(output_h5, tx_depths[i])
        arg_list.append(args)

    nCpus = mpi.cpu_count()
    p = mpi.Pool() # 5??
    print(arg_list)
    with p:
        p.map(solver, arg_list)