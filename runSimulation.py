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

if len(sys.argv) != 3:
    print('error: should have format: python runSimulation.py <input-file.txt> <output_file.h5>')
    sys.exit()

fname_in = str(sys.argv[1])
fname_out = str(sys.argv[2])

#Check if File Exists
if os.path.isfile(fname_out) == True:
    print('Warning, you are about to overwrite existing simulation data: ', fname_out, '\nProceed? [yes or no]')
    yes = {'yes', 'y'}
    no = {'no', 'n'}
    choice = input().lower()
    if choice in yes:
        print('Proceed')
        os.remove(fname_out)
    elif choice in no:
        print('Abort')
        sys.exit()
    else:
        print("Error, please start again and enter (yes / y) or (no / n)")
        sys.exit()

output_hdf = h5py.File(fname_out, 'w')

simul_start = datetime.datetime.now()
output_hdf.attrs["StartTime"] = simul_start.strftime("%Y\%m\%d %H:%M:%S") #time that simulation starts

iceDepth = util.select_variable("iceDepth", fname_in) #Ice Depth -> Max Depth (defined as z > 0) of Simulation (not including buffer)
iceLength = util.select_variable("iceLength", fname_in) #Ice Length -> Max Range (x) of simulation
airHeight = util.select_variable("airHeight", fname_in) #Air/Vacuum Height (defined as z < 0) of Simulation (not including buffer)
dx = util.select_variable("dx", fname_in) #Range Resolution
dz = util.select_variable("dz", fname_in) #Depth Resolution

#simul_mode = 'backwards_solver'
simul_mode = util.select_string("simul_mode", fname_in) #Selects simulation mode: choices: 1D, 2D, backwards_solver
print(simul_mode)
#Save to HDF File
output_hdf.attrs["iceDepth"] = iceDepth
output_hdf.attrs["iceLength"] = iceLength
output_hdf.attrs["airHeight"] = airHeight
output_hdf.attrs["dx"] = dx
output_hdf.attrs["dz"] = dz

tx_depths = util.select_range("tx_depths", fname_in) #list array containing the depth of each transmitter in the simulation run
rx_depths = util.select_range("rx_depths", fname_in) #list array containing the depth of each receiver in simulation
rx_ranges = util.select_range("rx_ranges", fname_in) #list array containing the range of each recevier
nTX = len(tx_depths)
output_hdf.attrs["nTransmitters"] = nTX

nRX_depths = len(rx_depths)
nRX_ranges = len(rx_ranges)
rxList= []
rxArray = np.ones((nRX_ranges, nRX_depths, 2))

output_hdf.create_dataset("tx_depths", data = tx_depths)
output_hdf.create_dataset("rx_depths", data = rx_depths)
output_hdf.create_dataset("rx_ranges", data = rx_ranges)

for i in range(len(rx_ranges)):
    for j in range(len(rx_depths)):
        xRX = rx_ranges[i] #Range of Receiver
        zRX = rx_depths[j] #Depth of Receiver
        rx_ij = rx(xRX, zRX)
        rxList.append(rx_ij)
        rxArray[i,j,0] = xRX
        rxArray[i,j,1] = zRX
nRX = len(rxList)
output_hdf.attrs["nReceivers"] = nRX

method = util.select_string("method", fname_in)

nCrevass = util.count_objects("crevass", fname_in)
nAquifer = util.count_objects("aquifer", fname_in)
nMeteors = util.count_objects("meteor", fname_in)

def nProf(x, z):
    n_material = epsilon.eps2m(epsilon.pure_ice(x,z))
    return n_material

if method == "func":
    profile_str = util.select_string("profile", fname_in)

    if profile_str == "enceladus_environ":
        snow_depth0 = util.select_variable("snow_depth", fname_in)
        crevass_list0 = []
        aquifer_list0 = []
        meteor_list0 = []
        if nCrevass > 0:
            crevass_list0 = util.select_crevass(fname_in)
        if nAquifer > 0:
            aquifer_list0 = util.select_aquifer(fname_in)
        if nMeteors > 0:
            meteor_list0 = util.select_meteor(fname_in)
        if simul_mode == "backwards_solver":
            def nProf(x,z):
                return epsilon.enceladus_environ(x, z, snow_depth = snow_depth0, crevass_list = crevass_list0, aquifer_list = aquifer_list0, meteor_list = meteor_list0)
        elif simul_mode == "2D":
            def nProf(x,z):
                return epsilon.enceladus_environ(x, z, snow_depth = snow_depth0)
        elif simul_mode == "1D":
            def nProf(z):
                return epsilon.enceladus_2layer(z, snow_depth=snow_depth0)
    elif profile_str == 'pure_ice':
        def nProf(x,z):
            return epsilon.pure_ice(x,z)
    elif profile_str == "south_pole":
        def nProf(z):
            return epsilon.south_pole(z)
    else:
        def nProf(x,z):
            return epsilon.pure_ice(x,z)

output_hdf.attrs["nProf"] = profile_str

#Next Step -> Set Pulse
#TODO -> Decide on Unit convention, ns/GHz or s/Hz??
Amplitude = util.select_variable("Amplitude", fname_in)
freqCentral = util.select_variable("freqCentral", fname_in) #Central Frequency
freqLP= util.select_variable("freqLP", fname_in) #Low Pass frequency (maximum frequency of pulse)
freqHP = util.select_variable("freqHP", fname_in) #High Pass Frequency (minimum frequency of pulse)

freqMin = util.select_variable("freqMin", fname_in) #Minimum Frequency of Simulation -> should include 95% of the gaussian width-> roughly 2*freq_min
freqMax = util.select_variable("freqMax", fname_in) #Maximum Frequency of SImulation

freqSample = util.select_variable("freqSample", fname_in) #Sampling Frequency
freqNyquist = freqSample/2 #Nyquist Frequency -> maximum frequency resolution of FFT -> needed for setting time resolution dt / sampling rate
tCentral = util.select_variable("tCentral", fname_in) #Central time of pulse -> i.e. time of pulse maximum
tSample = util.select_variable("tSample", fname_in)
dt = 1/freqSample
tspace = np.arange(0, tSample, dt) #Time Domain of Simulation
nSamples = len(tspace)
fftfreq_space = np.fft.fftfreq(nSamples, dt) #Full frequency space of FFT

#Save Pulse Settings
#TODO: Review, should I save pulse to each transmitter (could be useful in phased array) -> or save to whole simulation
output_hdf.attrs["Amplitude"] = Amplitude
output_hdf.attrs["freqCentral"] = freqCentral
output_hdf.attrs["freqLP"] = freqLP
output_hdf.attrs["freqHP"] = freqHP
output_hdf.attrs["freqSample"] = freqSample
output_hdf.attrs["freqNyquist"] = freqNyquist
output_hdf.attrs["tCentral"] = tCentral
output_hdf.attrs["tSample"] = tSample
output_hdf.attrs["tCentral"] = tCentral
output_hdf.attrs["dt"] = dt
output_hdf.attrs["nSamples"] = nSamples

for ind_tx in range(nTX):
    sourceDepth = tx_depths[ind_tx]
    print('Simulation Run:', ind_tx)

    tx_label = "tx-" + str(ind_tx)
    tx_hdf = output_hdf.create_group(tx_label)
    tx_hdf.attrs["sourceDepth"] = sourceDepth

    #Set Simulation
    print("Set Simulation Geometry")
    sim = ppp.paraProp(iceLength, iceDepth, dx, dz, refDepth=sourceDepth)

    ### useful arrays for plotting ###
    z = sim.get_z()
    x = sim.get_x()

    # Permittivity Profile
    print("Set Refractive Index Profile: ", profile_str)
    tstart_n2 = time.time()
    if simul_mode == "1D":
        sim.set_n("func", nFunc=nProf)
    elif simul_mode == "2D" or simul_mode == "backwards_solver":
        sim.set_n2("func", nFunc=nProf)
    tend_n2 = time.time()
    print("n profile set, duration = ", tend_n2 - tstart_n2)

    print("Source Depth: ", sourceDepth)
    sim.set_dipole_source_profile(freqCentral, sourceDepth)

    #Set Pulse():
    print("Set Pulse Parameters")
    impulse = util.make_pulse(tspace, tCentral, Amplitude, freqCentral)
    tx_pulse = util.normToMax(util.butterBandpassFilter(impulse, freqHP, freqLP, freqSample, 4))
    tx_hdf.create_dataset("signalPulse", data = tx_pulse)
    tx_hdf.create_dataset("signalSpectrum", data = np.fft.fft(tx_pulse))
    sim.set_td_source_signal(tx_pulse, dt)

    print("Solve PE...")

    tstart_solver = time.time()
    #rxList_out = backwards_solver(sim, rxList, freqMin, freqMax, 0.1)
    if simul_mode == "backwards_solver":
        sim.backwards_solver_2way(rxList, freqMin, freqMax, 1, 0.1)
    elif simul_mode == "2D":
        sim.do_solver2(rxList, freqMin, freqMax, 1)
    elif simul_mode == "1D":
        sim.do_solver(rxList)
    else:
        print("Warning, invalid simul_mode, please set #simul_mode backwards_solver, 2D or 1D in ", fname_in)
        sys.exit()
    tend_solver = time.time()
    tduration_solver = tend_solver - tstart_solver
    print("Solution complete:, duration: ", tduration_solver)

    for ind_rx in range(nRX):
        rx = rxList[ind_rx]

        rx_label = "rx-" + str(ind_rx)
        rx_hdf = tx_hdf.create_group(rx_label)

        rx_hdf.attrs["Range"] = rx.x
        rx_hdf.attrs["Depth"] = rx.z

        rx_signal = rx.get_signal()
        rx_spectrum = rx.get_spectrum()
        rx_hdf.create_dataset("rxPulse", data=rx_signal)
        rx_hdf.create_dataset("rxSpectrum", data=rx_spectrum)
    print("saving date to file, proceed with next source \n")

simul_end = datetime.datetime.now()
output_hdf.attrs["EndTime"] = simul_end.strftime("%Y\%m\%d %H:%M:%S") #time that simulation starts

duration = simul_end - simul_start
duration_in_s = duration.total_seconds()
output_hdf.attrs["Duration"] = duration_in_s

output_hdf.close()