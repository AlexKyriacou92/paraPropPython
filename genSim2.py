import numpy as np
from permittivity import *
from geometry import triangle, circle
import paraPropPython as ppp
from paraPropPython import receiver as rx
import util
from backwardsSolver import backwards_solver
from matplotlib import pyplot as pl
import permittivity as epsilon
import time

import h5py
import sys
import datetime
import os

if len(sys.argv) != 3:
    print('error: should have format: python runSimulation.py <input-file.txt> <output_file.h5>')
    sys.exit()

fname_in = str(sys.argv[1]) #input file
fname_out = str(sys.argv[2]) #output files

fname_hdf = fname_out + '.h5'
#Check if File Exists

if os.path.isfile(fname_hdf) == True:
    print('Warning, you are about to overwrite existing simulation data: ', fname_hdf, '\nProceed? [yes or no]')
    yes = {'yes', 'y'}
    no = {'no', 'n'}
    choice = input().lower()
    if choice in yes:
        print('Proceed')
        os.remove(fname_hdf)
    elif choice in no:
        print('Abort')
        sys.exit()
    else:
        print("Error, please start again and enter (yes / y) or (no / n)")
        sys.exit()


output_hdf = h5py.File(fname_hdf, 'w')

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
output_hdf.attrs["mode"] = simul_mode

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
        #rxList.append(rx_ij)
        rxArray[i,j,0] = xRX
        rxArray[i,j,1] = zRX
nRX = len(rxList)

output_hdf.create_dataset("rxArray", data=rxArray)
output_hdf.attrs["nReceivers"] = nRX

method = util.select_string("method", fname_in)

nCrevass = util.count_objects("crevass", fname_in)
nAquifer = util.count_objects("aquifer", fname_in)
nMeteors = util.count_objects("meteor", fname_in)

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

df = 1/tSample

tspace = np.arange(0, tSample, dt) #Time Domain of Simulation
nSamples = len(tspace)

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

impulse = util.make_pulse(tspace, tCentral, Amplitude, freqCentral)
tx_pulse = util.normToMax(util.butterBandpassFilter(impulse, freqHP, freqLP, freqSample, 4))
output_hdf.create_dataset('signalPulse', data=tx_pulse)
output_hdf.create_dataset('signalSpectrum', data=np.fft.fft(tx_pulse))

#TODO: Create Geometry File


def nProf(x, z):
    n_material = epsilon.eps2m(epsilon.pure_ice(x,z))
    return n_material

if method == "func":
    profile_str = util.select_string("profile", fname_in)

    if profile_str == "enceladus_environ":
        print("selected: ", profile_str)
        snow_depth0 = util.select_variable("snow_depth", fname_in)
        print('snow layer:', snow_depth0, 'm')
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
            print('backwards solver')
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
print(profile_str)
x = np.arange(0, iceLength+dx, dx)
xNum = len(x)

z = np.arange(-airHeight, iceDepth + dz, dz)

filterDepth=100
zFull = np.arange(-(airHeight + filterDepth), iceDepth + filterDepth + dz, dz)
zNumFull = len(zFull)

if simul_mode == "2D" or simul_mode == "backwards_solver":
    print("method:", method)
    zNum_red = xNum * 10
    zRed = np.arange(-(airHeight + filterDepth), iceDepth + filterDepth + dz, dx)
    zNum_red = len(zRed)

    n2Mat_red = np.ones((xNum, zNum_red), dtype='complex')

    output_mat = fname_out + '-nProf.npy'
    n2Mat = util.create_memmap(output_mat, (xNum, zNumFull))
    print(output_mat)
    print('create geometry')
    for i in range(xNum):
        for j in range(zNum_red):
            n2Mat_red[i, j] = nProf(x[i], zRed[j])

    for i in range(xNum):
        n_prof = n2Mat_red[i]
        n_interp = np.interp(zFull, zRed, n_prof)
        n2Mat[i] = n_interp

    pl.figure(figsize=(10,10), dpi = 120)
    pmesh = pl.imshow(np.transpose(epsilon.m2eps(n2Mat_red).real),aspect='auto', extent=(x[0], x[-1], zFull[-1], zFull[0]))
    pl.xlabel('Range x [m]')
    pl.ylabel('Depth z [m]')
    pl.ylim(z[-1],z[0])
    cbar = pl.colorbar(pmesh)
    cbar.set_label(r"Permittivity (real) $\epsilon_{r}^{'}$")
    pl.savefig(fname_out + '-nref-real.png')
    #pl.show()
    pl.close()

elif simul_mode == "1D":
    n1vector = np.zeros(zNumFull, dtype='complex')
    output_vector = fname_out + '-nProf.npy'
    for i in range(zNumFull):
        n1vector[i] = nProf(zFull[i])
    np.save(output_vector, n1vector)

#Create Numpy File:
fname_npy = fname_out + '.npy'
dimensions = (nTX, nRX_ranges, nRX_depths, nSamples)
npy_memmap = util.create_memmap(fname_npy, dimensions)

output_hdf.close()

#create -> a job list for each source

jobfile = fname_out + "-jobs.txt"
job_list = open(jobfile, "w+")
for itx in range(nTX):
    line = "python runSolver2.py " + fname_out + " " + str(tx_depths[itx]) + "\n"
    job_list.write(line)
job_list.close()
shfile = fname_out + "-submitjobs"
makescript = "python pleiades-job-sbatch.py " + jobfile + " " + shfile
os.system(makescript)
"""
script_list = []

for i_tx in range(nTX):
    jobname = "src=" + str(tx_depths[i_tx])
    jobfname = path2jobs + "/" + jobname + '.txt'

    job_file = open(jobfname, 'w+')
    #freq_list = np.arange(freqHP, freqLP, df)
    for j_freq in range(len(freq_list)):
        line = "python runSolver.py " + fname_out + " " + str(freq_list[j_freq]) + " " + str(tx_depths[i_tx]) + "\n"
        job_file.write(line)
    job_file.close()
    #Make a job submission script for pleaides
    shfile = path2jobs + "/" + jobname
    makescript = "python pleiades-job-sbatch.py " + jobfname + " " + shfile
    os.system(makescript)
    os.remove(jobfname)

"""