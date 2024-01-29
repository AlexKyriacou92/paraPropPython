import os.path
import sys
import numpy as np
import time
import datetime
import h5py
import configparser
import pytz
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

#FILE DESCRIPTION -> USE TXT FILES TO WRITE SPECTRUM AT RXs

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
fname_config = 'config_summit_1km.txt'
fname_nProf = sys.argv[1]
fname_CFM = 'CFMresults.hdf5'

config_in = configparser.ConfigParser()
config_in.read(fname_config)
#cfm_config = config_in['CFM']
input_config = config_in['INPUT']

date_arr = get_dates(fname_CFM)


from data import create_transmitter_array_from_file
#STEP 1 -> Create File
#Make Directory                                                                                                                                                                                                                                                                                                               
dir_sim = 'CFM_files_ku/large_scale_pulse_simple_1km'
if os.path.isdir(dir_sim) == False:
    os.system('mkdir ' + dir_sim)
dir_sim_path = dir_sim + '/'

sim_prefix = fname_nProf[:-4]
sim_name = sim_prefix
fname_body = sim_prefix
fname_hdf0 = fname_body + '.h5'
fname_npy0 = fname_body + '.npy'

n_profile_example, z_profile = util.get_profile_from_file(fname_nProf)

fname_hdf = dir_sim_path + fname_hdf0

with h5py.File(fname_hdf, 'w') as hdf_in:
    hdf_in.attrs['name'] = sim_name


txList = create_transmitter_array_from_file(fname_config)
nTx = len(txList)
tx_signal_in = create_spectrum(fname_config=fname_config,
                               nprof_data=n_profile_example, zprof_data=z_profile,
                               fname_output_h5=fname_hdf)


tx_pulse_in = tx_signal_in.pulse
tx_spectrum_in = tx_signal_in.spectrum_plus
freq_plus = tx_signal_in.freq_plus
tspace = tx_signal_in.tspace
nSamples = tx_signal_in.nSamples

rxList = create_rxList_from_file(fname_config)
nRx = len(rxList)

ii_tx = 0
z_tx = txList[ii_tx]

#The Script will dispatch jobs for the Frequnecy Range (Min to Max, i.e. 50 MHz to 450 MHz)
freqMin = tx_signal_in.freqMin
freqMax = tx_signal_in.freqMax
ii_min = util.findNearest(freq_plus, freqMin)
ii_max = util.findNearest(freq_plus, freqMax)
# STEP 2 -> SEND OUT SCRIPTS

#Write the Name of the File
fname_list = fname_body + '_list.txt'

fout_list = open(dir_sim_path + fname_list, 'w')
fout_list.write(dir_sim_path+ '\t' + fname_hdf0 + '\t' + fname_npy0 +'\n')
fout_list.write(str(nTx) + '\t' + str(nRx) + '\t' + str(nSamples) + '\n')
fout_list.write('ID_Freq\tFreq_GHz\tfname_npy\n')

for ii_freq in range(ii_min, ii_max):
    freq_ii = freq_plus[ii_freq]
    fname_txt_i = fname_body + '_' + str(ii_freq) + '.txt'
    print('create job for f = ', freq_ii*1e3, ' MHz')

    line = str(ii_freq) + '\t' + str(round(freq_ii,3)) + '\t' + fname_txt_i + '\n'
    fout_list.write(line)

    fname_npy_i = dir_sim_path + fname_txt_i
    util.create_memmap2(fname_npy_i, dimensions=(nTx, nRx), data_type='complex')

    cmd = 'python runSim_ascan_rx_from_txt.py ' + fname_config + ' '
    cmd += fname_npy_i + ' ' + fname_hdf + ' ' + fname_nProf + ' '
    cmd += str(ii_freq) + ' ' + str(ii_tx)

    suffix = 'fid_' + str(int(freq_ii*1e3))
    jobname = dir_sim_path + suffix
    fname_sh_in = 'sim_CFM_' + suffix + '.sh'

    fname_sh_out0 = 'sim_CFM_' + sim_name + '_' + suffix + '.out'
    fname_sh_out = dir_sim_path + fname_sh_out0

    make_job(fname_shell=fname_sh_in, fname_outfile=fname_sh_out, jobname=jobname, command=cmd)
    submit_job(fname_sh_in)
    os.system('rm ' + fname_sh_in)
fout_list.close()
