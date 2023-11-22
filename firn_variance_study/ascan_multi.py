import os.path
import sys
import numpy as np
import time
import datetime
import h5py
import configparser
from numpy import exp, log
import matplotlib.pyplot as pl
from ku_scripting import *


sys.path.append('../')
from paraPropPython import paraProp as ppp
import receiver as rx
from transmitter import tx_signal
from data import create_sim, create_rxList_from_file, create_tx_signal, create_transmitter_array
from data import create_hdf_bscan, bscan_rxList, create_hdf_FT, create_tx_signal_from_file
from data import create_ascan_hdf, ascan
import util

def create_spectrum(fname_config, nprof_data, zprof_data, fname_output_h5):
    tx_signal_out = create_tx_signal(fname_config)
    rxList = create_rxList_from_file(fname_config)
    txList = create_transmitter_array(fname_config)

    nSamples = tx_signal_out.nSamples
    nRX = len(rxList)
    nTX = len(txList)

    # Create TX_Signal - Impulse
    tx_signal_out = create_tx_signal(fname_config)
    tx_signal_out.set_impulse()
    f_centre = tx_signal_out.frequency
    bandwidth = tx_signal_out.bandwidth
    fmin = f_centre - bandwidth/2
    fmax = f_centre + bandwidth/2
    tx_signal_out.apply_bandpass(fmin=fmin, fmax=fmax)
    tx_signal_out.add_gaussian_noise()

    #util.create_memmap2(fname_output_npy,(nTX, nRX, nSamples))
    hdf_ascan = create_ascan_hdf(fname_config=fname_config,
                                 tx_signal=tx_signal_out,
                                 nprof_data=nprof_data,
                                 zprof_data=zprof_data,
                                 fname_output=fname_output_h5)
    hdf_ascan.close()
    return tx_signal_out

def run_scan_impulse(fname_config, fname_output, nprofile_data, zprofile_data, ii_freq, jj_tx):
    tx_signal_out = create_tx_signal(fname_config)
    rxList = create_rxList_from_file(fname_config)
    txList = create_transmitter_array(fname_config)

    nSamples = tx_signal_out.nSamples
    nRX = len(rxList)
    nTX = len(txList)

    tx_signal.set_impulse()
    f_centre = tx_signal.frequency
    bandwidth = tx_signal.bandwidth
    fmin = f_centre - bandwidth/2
    fmax = f_centre + bandwidth/2
    tx_signal.apply_bandpass(fmin=fmin, fmax=fmax)
    tx_signal.add_gaussian_noise()

    spectrum = tx_signal.get_spectrum()
    freq_space = tx_signal.get_freq_space()

    amp_select = spectrum[ii_freq]
    freq_select = freq_space[ii_freq]
    z_tx = txList[jj_tx]

    sim = create_sim(fname_config)
    sim.set_n(nVec=nprofile_data, zVec=zprofile_data)  # Set Refractive Index Profile
    sim.set_dipole_source_profile(centerFreq=freq_select, depth=z_tx, A=amp_select)  # Set Source Profile
    sim.set_cw_source_signal(freq_select)
    sim.do_solver()

    with np.load(fname_output, 'w+') as spectrum_npy:
        for kk_rx in range(nRX):
            rx_kk = rxList[kk_rx]
            x_rx = rx_kk.x
            z_rx = rx_kk.z
            spectrum_npy[jj_tx, kk_rx, ii_freq] = sim.get_field(x0=x_rx,z0=z_rx)

def save_spectrum(fname_output_npy, fname_output_hdf):
    spectrum_npy = np.load(fname_output_npy)
    ascan_out = ascan()
    ascan_out.load_from_hdf(fname_output_hdf)
    ascan_out.load_spectrum(spectrum_npy)

def get_data(fname_in, input_var='density', jj = 0):
    with  h5py.File(fname_in, 'r') as input_hdf:
        z_arr = np.array(input_hdf['depth'])
        data_matrix = np.array(input_hdf[input_var])
    data_arr = data_matrix[jj]
    data_arr = data_arr[1:]
    z_arr = z_arr[1:]
    return data_arr, z_arr

def get_dates(fname_in):
    with  h5py.File(fname_in, 'r') as input_hdf:
        data_matrix = np.array(input_hdf['density'])
    date_arr = data_matrix[1:,0]
    return date_arr

def select_date(date_arr, year, month):
    date_num = float(year) + (month)/12.
    ii_num = util.findNearest(date_arr, date_num)
    return ii_num
# Start With One Spectrum Point

#INPUTS
fname_config = 'config_summit_large_scale_cfm.txt'
fname_nProf = 'nProf_CFM_deep2.h5'
fname_CFM = 'CFMresults.hdf5'

config_in = configparser.ConfigParser()
config_in.read(fname_config)
cfm_config = config_in['CFM']
input_config = config_in['INPUT']

sim_prefix = input_config['config_name']
year_example = int(cfm_config['year'])
month_example = float(cfm_config['month'])

nProf_hdf = h5py.File(fname_nProf, 'r')
nProf_matrix = np.array(nProf_hdf['n_profile_matrix'])
z_profile = np.array(nProf_hdf['z_profile'])
nProf_hdf.close()

date_arr = get_dates(fname_CFM)
ii_date = select_date(date_arr, year_example, month_example)

n_profile_example = nProf_matrix[ii_date]

from data import create_transmitter_array_from_file
#STEP 1 -> Create File
#Make Directory                                                                                                                                                                                                                                                                                                               
dir_sim = 'CFM_files_ku/large_scale_pulse'
if os.path.isdir(dir_sim) == False:
    os.system('mkdir ' + dir_sim)
dir_sim_path = dir_sim + '/'


fname_body = sim_prefix + '_' + str(year_example) + '_' + str(month_example).zfill(2)
fname_hdf0 = fname_body + '.h5'
fname_npy0 = fname_body + '.npy'

fname_hdf = dir_sim_path + fname_hdf0

sim_name = sim_prefix + '_' + str(year_example) + ' ' + str(month_example)
with h5py.File(fname_hdf, 'w') as hdf_in:
    hdf_in.attrs['name'] = sim_name
    hdf_in.attrs['year'] = year_example
    hdf_in.attrs['month'] = month_example

txList = create_transmitter_array_from_file(fname_config)
nTx = len(txList)
tx_signal_in = create_spectrum(fname_config=fname_config,
                               nprof_data=n_profile_example, zprof_data=z_profile,
                               fname_output_h5=fname_hdf)


tx_pulse_in = tx_signal_in.pulse
tx_spectrum_in = tx_signal_in.get_spectrum()
freq_space = tx_signal_in.get_freq_space()
tspace = tx_signal_in.tspace
nSamples = tx_signal_in.nSamples

rxList = create_rxList_from_file(fname_config)
nRx = len(rxList)

ii_tx = 0
z_tx = txList[ii_tx]

#The Script will dispatch jobs for the Frequnecy Range (Min to Max, i.e. 50 MHz to 450 MHz)
freqMin = tx_signal_in.freqMin
freqMax = tx_signal_in.freqMax
ii_min = util.findNearest(freq_space, freqMin)
ii_max = util.findNearest(freq_space, freqMax)
# STEP 2 -> SEND OUT SCRIPTS

#Write the Name of the File
fname_list = fname_body + '_list.txt'

fout_list = open(dir_sim_path + fname_list, 'w')
fout_list.write(dir_sim_path+ '\t' + fname_hdf0 + '\t' + fname_npy0 +'\n')
fout_list.write(str(nTx) + '\t' + str(nRx) + '\t' + str(nSamples) + '\n')
fout_list.write('ID_Freq\tFreq_GHz\tfname_npy\n')

for ii_freq in range(ii_min, ii_max):
    freq_ii = freq_space[ii_freq]
    fname_npy_i = fname_body + '_' + str(ii_freq) + '.npy'
    print('create job for f = ', freq_ii*1e3, ' MHz')

    line = str(ii_freq) + '\t' + str(round(freq_ii,3)) + '\t' + fname_npy_i + '\n'
    fout_list.write(line)

    fname_npy_i = dir_sim_path + fname_npy_i
    util.create_memmap2(fname_npy_i, dimensions=(nTx, nRx), data_type='complex')

    cmd = 'python runSim_ascan_rx.py ' + fname_config + ' '
    cmd += fname_npy_i + ' ' + fname_hdf + ' ' + fname_nProf + ' '
    cmd += str(ii_date) + ' ' + str(ii_freq) + ' ' + str(ii_tx)

    suffix = 'fid_' + str(int(freq_ii*1e3))
    jobname = dir_sim_path + suffix
    fname_sh_in = 'sim_CFM_' + suffix + '.sh'

    fname_sh_out0 = 'sim_CFM_' + suffix + '.out'
    fname_sh_out = dir_sim_path + fname_sh_out0

    make_job(fname_shell=fname_sh_in, fname_outfile=fname_sh_out, jobname=jobname, command=cmd)
    submit_job(fname_sh_in)
    os.system('rm ' + fname_sh_in)
fout_list.close()