import os.path
import sys
import numpy as np
import time
import datetime
import h5py
import configparser
from numpy import exp, log
import matplotlib.pyplot as pl
import pytz

import math
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

def get_ym(date_arr, ii):
    date_num = date_arr[ii]
    year = math.floor(date_num)
    month = round(date_num%year)
    return year, month
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
#year_example = int(cfm_config['year'])
#month_example = float(cfm_config['month'])

nProf_hdf = h5py.File(fname_nProf, 'r')
nProf_matrix = np.array(nProf_hdf['n_profile_matrix'])
z_profile = np.array(nProf_hdf['z_profile'])
nProf_hdf.close()

date_arr = get_dates(fname_CFM)
#ii_date = select_date(date_arr, year_example, month_example)


from data import create_transmitter_array_from_file
#STEP 1 -> Create File
#Make Directory                                                                                                                                                                                                                                                                                                               
dir_sim = 'CFM_files_ku/large_scale_pulse_1500m'
if os.path.isdir(dir_sim) == False:
    os.system('mkdir ' + dir_sim)
dir_sim_path = dir_sim + '/'

nDates = len(date_arr)

print(nDates, nDates//5)

N = 5
M = nDates // 5
t_sleep = 60 * 60 * 2 # 2 hours

for jj in range(1):

    ii_start = jj*N
    ii_end = (jj+1)*N

    fname_overlist = 'ascan_script_' + str(jj).zfill(3) + '.txt'
    f_overlist = open(fname_overlist, 'w')


    for ii_date in range(ii_start, ii_end):
        date_num = date_arr[ii_date]
        year_example, month_example = get_ym(date_arr, ii_date)
        n_profile_example = nProf_matrix[ii_date]

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

        #tx_signal_in = create_tx_signal_from_file(fname_config)
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
            fname_txt_i = fname_body + '_' + str(ii_freq) + '.txt'
            print('create job for f = ', freq_ii*1e3, ' MHz')

            line = str(ii_freq) + '\t' + str(round(freq_ii,3)) + '\t' + fname_txt_i + '\n'
            fout_list.write(line)

            fname_npy_i = dir_sim_path + fname_txt_i
            util.create_memmap2(fname_npy_i, dimensions=(nTx, nRx), data_type='complex')

            cmd = 'python runSim_ascan_rx.py ' + fname_config + ' '
            cmd += fname_npy_i + ' ' + fname_hdf + ' ' + fname_nProf + ' '
            cmd += str(ii_date) + ' ' + str(ii_freq) + ' ' + str(ii_tx)

            suffix = 'fid_' + str(int(freq_ii*1e3))
            jobname = dir_sim_path + suffix
            fname_sh_in = 'sim_CFM_' + suffix + '.sh'

            fname_sh_out0 = 'sim_CFM_' + str(year_example) + '_' + str(month_example) + '_' + suffix + '.out'
            fname_sh_out = dir_sim_path + fname_sh_out0

            make_job(fname_shell=fname_sh_in, fname_outfile=fname_sh_out, jobname=jobname, command=cmd)
            #submit_job(fname_sh_in)
            os.system('rm ' + fname_sh_in)

        line_overlist = fname_list + '\n'
        f_overlist.write(line_overlist)

        fout_list.close()
    f_overlist.close()
    print('Time To Wait')
    now_cst = datetime.datetime.now(tz=pytz.timezone('US/Indiana-Starke'))
    now_cet = datetime.datetime.now(tz=pytz.timezone('CET'))
    now_act = datetime.datetime.now(tz=pytz.timezone('Australia/South'))

    tstamp_now = now_cet.timestamp()
    tstamp_future = tstamp_now + t_sleep
    print('Current Time (Lawrence, KS, USA):', now_cst.strftime('%Y.%m.%d %H:%M:%S'))
    print('Current Time (Wuppertal, NRW, Germany):', now_cet.strftime('%Y.%m.%d %H:%M:%S'))
    print('Current Time (Adelaide, SA, Australia):', now_act.strftime('%Y.%m.%d %H:%M:%S'))

    future_cst = datetime.datetime.fromtimestamp(tstamp_future, tz=pytz.timezone('US/Indiana-Starke'))
    future_cet = datetime.datetime.fromtimestamp(tstamp_future, tz=pytz.timezone('CET'))
    future_act = datetime.datetime.fromtimestamp(tstamp_future, tz=pytz.timezone('Australia/South'))

    print('Time of Completion (Lawrence, KS, USA):', future_cst.strftime('%Y.%m.%d %H:%M:%S'))
    print('Time of Completion (Wuppertal, NRW, Germany):', future_cet.strftime('%Y.%m.%d %H:%M:%S'))
    print('Time of Completion (Adelaide, SA, Australia):', future_act.strftime('%Y.%m.%d %H:%M:%S'))

    #future_act

    time.sleep(t_sleep)
    #Save Txt Files to Numpy Array

    print('Saving Txt Files to Numpy Arrays')
    f_overlist2 = open(fname_overlist, 'r')
    for line in f_overlist2:
        fname_list = line
        os.system('python add_spectrum_to_hdf.py ' + fname_list)
    f_overlist2.close()