import os.path
import sys
import numpy as np
import time
import datetime
import h5py
import configparser
from numpy import exp, log
import matplotlib.pyplot as pl
from makeDepthScan import depth_scan_impulse_smooth
from ku_scripting import *

sys.path.append('../')
from paraPropPython import paraProp as ppp
import receiver as rx
from transmitter import tx_signal
from data import create_sim, create_rxList_from_file, create_tx_signal, create_transmitter_array
from data import create_hdf_bscan, bscan_rxList, create_hdf_FT
from data import create_ascan_hdf, ascan

import util
from data import create_transmitter_array_from_file

from Askaryan_Signal import create_pulse as create_pulse_askaryan, TeV

def create_pulse_askaryan_config(fname_config):
    config = configparser.ConfigParser()
    config.read(fname_config)
    ask_config = config['ASKARYAN']

    dtheta_v = float(ask_config['dtheta_v'])
    E_nu_eV = float(ask_config['showerEnergy'])
    E_nu_TeV = E_nu_eV/TeV
    R_alpha = float(ask_config['attenuationEquivalent'])
    tx_signal_out = create_tx_signal(fname_config)
    tspace = tx_signal_out.tspace
    tmax = tx_signal_out.tmax
    t_center = tx_signal_out.t_centre

    pulse_ask_out, tspace_ask = create_pulse_askaryan(Esh=E_nu_TeV, dtheta_v=dtheta_v, R_alpha=R_alpha,
                                                      t_min=-1*t_center/1e3, t_max=(tmax-t_center)/1e3,
                                                      fs=tx_signal_out.fsample*1e3)
    pulse_ask_interp = np.interp(tspace, tspace_ask, pulse_ask_out)
    tx_signal_out.set_pulse(pulse_ask_interp, tspace)
    return tx_signal_out


def create_spectrum(fname_config, nprof_data, zprof_data, fname_output_h5):
    tx_signal_out = create_pulse_askaryan_config(fname_config)
    hdf_ascan = create_ascan_hdf(fname_config=fname_config,
                                 tx_signal=tx_signal_out,
                                 nprof_data=nprof_data,
                                 zprof_data=zprof_data,
                                 fname_output=fname_output_h5)
    hdf_ascan.close()
    return tx_signal_out

if len(sys.argv) == 2:
    fname_config = sys.argv[1]
else:
    print('wrong arg number', len(sys.argv))
    print('Enter: python ', sys.argv[0], ' <config_file.txt>')
    sys.exit()

config = configparser.ConfigParser()
config.read(fname_config)
fname_nprof = config['REFRACTIVE_INDEX']['fname_profile']

nprof_data, zprof_data = util.get_profile_from_file(fname_nprof)
dir_sim = config['OUTPUT']['path2output']
if os.path.isdir(dir_sim) == False:
    os.system('mkdir ' + dir_sim)
dir_sim_path = dir_sim + '/'


sim_prefix = os.path.basename(fname_nprof)
sim_prefix = sim_prefix[:-4]
sim_name = sim_prefix
fname_body = sim_prefix
fname_hdf0 = fname_body + '.h5'
fname_npy0 = fname_body + '.npy'

fname_hdf = dir_sim_path + fname_hdf0

txList = create_transmitter_array_from_file(fname_config)
nTx = len(txList)
tx_signal_in = create_spectrum(fname_config=fname_config,
                               nprof_data=nprof_data, zprof_data=zprof_data,
                               fname_output_h5=fname_hdf)

tx_pulse_in = tx_signal_in.pulse
tx_spectrum_in = tx_signal_in.spectrum_plus

freq_plus = tx_signal_in.freq_plus
tspace = tx_signal_in.tspace
nSamples = tx_signal_in.nSamples

rxList = create_rxList_from_file(fname_config)
nRx = len(rxList)


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
fout_list.write('ID_TX\tID_Freq\tFreq_GHz\tfname_npy\n')

for ii_tx in range(nTx):
    z_tx = txList[ii_tx]
    for ii_freq in range(ii_min, ii_max):
        freq_ii = freq_plus[ii_freq]
        fname_txt_i = fname_body + '_' + str(ii_tx).zfill(2) + '_' + str(ii_freq) + '.txt'
        fname_txt_i = fname_txt_i
        fname_txt_path = os.path.join(dir_sim_path, fname_txt_i)
        print('create job for, z_tx = ', z_tx, ' m, f = ', freq_ii*1e3, ' MHz')

        line = str(ii_tx) + '\t' + str(ii_freq) + '\t' + str(round(freq_ii,3)) + '\t' + fname_txt_i + '\n'
        fout_list.write(line)

        #fname_npy_i = dir_sim_path + fname_txt_i
        #util.create_memmap2(fname_npy_i, dimensions=(nTx, nRx), data_type='complex')

        cmd = 'python runSim_ascan_rx_from_txt.py ' + fname_config + ' '
        cmd += fname_txt_path + ' ' + fname_hdf + ' ' + fname_nprof + ' '
        cmd += str(ii_freq) + ' ' + str(ii_tx)

        suffix = 'fid_' + str(ii_tx).zfill(2) + '_' + str(int(freq_ii*1e3))

        jobname = dir_sim_path + suffix
        fname_sh_in = 'sim_CFM_' + suffix + '.sh'

        fname_sh_out0 = 'sim_CFM_' + sim_name + '_' + suffix + '.out'
        fname_sh_out = dir_sim_path + fname_sh_out0

        make_job(fname_shell=fname_sh_in, fname_outfile=fname_sh_out, jobname=jobname, command=cmd)
        submit_job(fname_sh_in)
        os.system('rm ' + fname_sh_in)

fout_list.close()
'''
#Add Spectrrum to HDF file
fname_npy = os.path.join(dir_sim_path, fname_npy0)
ascan_sim = ascan()
ascan_sim.load_from_hdf(fname_hdf=fname_hdf)
ascan_sim.save_spectrum(fname_npy=fname_npy)
'''