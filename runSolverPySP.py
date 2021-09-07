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

    iceDepth = output_h5.attrs["iceDepth"] # 0
    args.append(iceDepth)
    iceLength = output_h5.attrs["iceLength"] # 1
    args.append(iceLength)

    airHeight0 = output_h5.attrs["airHeight"] # 2
    args.append(airHeight0)
    # output_hdf.attrs["airHeight"] = airHeight
    dx = output_h5.attrs["dx"] # 3
    args.append(dx)
    dz = output_h5.attrs["dz"] # 4
    args.append(dz)

    tx_depths = np.array(output_h5.get("tx_depths"))  # 5
    args.append(tx_depths)
    nTX = len(tx_depths)

    rx_depths = np.array(output_h5.get("rx_depths")) # 6
    args.append(rx_depths)

    nRX_depths = len(rx_depths)

    rx_ranges = np.array(output_h5.get("rx_ranges")) #7
    args.append(rx_ranges)
    nRX_ranges = len(rx_ranges)

    # Fill in blank
    # load geometry
    freqCentral = output_h5.attrs["freqCentral"]
    args.append(freqCentral) # 8

    mode = output_h5.attrs["mode"] # 9
    args.append(mode)
    print('mode:', mode)
    print(nProfile.shape)

    # tx pulse
    tx_pulse = np.array(output_h5.get("signalPulse")) # 10
    args.append(tx_pulse)
    # tx_spectrum = np.array(output_h5.get("signalSpectrum"))

    freqLP = output_h5.attrs["freqLP"] # 11
    args.append(freqLP)
    freqHP = output_h5.attrs["freqHP"] # 12
    args.append(freqHP)
    nSamples = output_h5.attrs["nSamples"] # 13
    args.append(nSamples)
    dt = output_h5.attrs["dt"] # 14
    args.append(dt)

    rxArray = np.array(output_h5.get("rxArray")) #15
    print(rxArray)
    args.append(rxArray)
    rxList = []
    ii = 0
    args.append(sourceDepth) #16
    # print('rxList = ', rxList)
    return args

def solver(args):
    iceDepth = args[0]
    print('iceDepth = ', iceDepth)
    iceLength = args[1]
    print('iceLength = ', iceLength)

    airHeight0 = args[2]
    print('airHeight = ', airHeight0)
    dx = args[3]
    print('dx =', dx)
    dz = args[4]
    print('dz =', dz)
    tx_depths = args[5]
    print('tx_depths = ', tx_depths)
    nTX = len(tx_depths)

    rx_depths = args[6]
    print('rx_depths =', rx_depths)
    nRX_depths = len(rx_depths)

    rx_ranges = args[7]
    print('rx_ranges = ', rx_ranges)
    nRX_ranges = len(rx_ranges)

    freqCentral = args[8]
    print('freq_central = ', freqCentral)
    mode = args[9]
    print('mode = ', mode)
    tx_pulse = args[10]
    print('tx_pulse =', tx_pulse)
    freqLP = args[11]
    freqHP = args[12]

    nSamples = args[13]
    print('nSamples =', nSamples)
    dt = args[14]
    print('dt = ', dt)
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
            spectrum = rx.spectrum()
            output_npy[ii_source, i, j, ii_freqHP:ii_freqLP] = spectrum[ii_freqHP:ii_freqLP]
            ii_rxList += 1

if __name__ == '__main__':
    tx_depths = np.array(output_h5.get("tx_depths"))
    nTX = output_h5.attrs["nTransmitters"]
    arg_list = []
    for i in range(nTX):
        args = get_args(output_h5, tx_depths[i])
        arg_list.append(args)
        tstart = time.time()
        print('source depth, Z_tx = ', tx_depths[i])
        solver(args)
        tend = time.time()