import os.path
import sys
from sys import exit
import numpy as np
import time
import datetime
import h5py
import configparser

from os import system
from ku_scripting import *

sys.path.append('../')

from data import create_sim, create_rxList_from_file, create_tx_signal, create_transmitter_array
from data import create_ascan_hdf, ascan
import util
from data import create_transmitter_array_from_file
from Askaryan_Signal import create_pulse as create_pulse_askaryan, TeV

'''
This Script is utilized for a study on how changes in firn density affects 
radio propagation from a deep RF signal

This Script is Designed to Run paraProp Distributed Cluster Simulations
for a list of n(z) profiles defined in a list of files and correspoding
to estimated n(z) from rho(z) for the Community Firn Model 

The process is complicated due to the reliance on 
a large number of other scripts defined in this directory and in the main paraProp code

All of the simulation data for each scneario is saved to HDF file, but the
spectrum for each RX and each freq are initially save to text files which then have
to be compiled and consolidated to the hdf file

I loop through every n(z) scenario -> run genSim_askaryan_config.py which runs th simulation
the geometry and signal proeprties is defined by a config file, the TX positions by an additional 'tx_list.txt' file
the RX positions by another txt file 'rx_list.txt' , and the n(z) profile are specified by txt files 
nProf_CFM_year_month.txt which are all listed in an additonal text file

As I loop through each simulation - the n(z) file and hdf file are saved to a log file

I apologize to any future readers for the extremely messy, ugly and needlessly complex code
I haven't yet worked out the balance between elegant/simple/readable code 
and code that gets what I need done in the here and now.
'''

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

if len(sys.argv) == 3:
    fname_config = sys.argv[1]
    fname_nprofile_all = sys.argv[2]
else:
    print('wrong arg number', len(sys.argv))
    print('Enter: python ', sys.argv[0], ' <config_file.txt> <nprof_l>')
    sys.exit()

config = configparser.ConfigParser()
config.read(fname_config)
dir_sim = config['OUTPUT']['path2output']
if os.path.isdir(dir_sim) == False:
    os.system('mkdir ' + dir_sim)
dir_sim_path = dir_sim + '/'

year_l = []
month_l = []
nprofile_list = []
path2profiles = 'ref_profiles_all'
with open(fname_nprofile_all) as all_profiles:
    for line in all_profiles:
        cols = line.split()
        year_l.append(int(cols[0]))
        month_l.append(int(cols[1]))
        nprofile_list.append(os.path.join(path2profiles, cols[2]))

nProfiles_all = len(nprofile_list)

t_wait = 10 # 10 Minutes

now = datetime.datetime.now()
datetime_str = now.strftime('%y%m%d_%H%M%S')
fname_config_new = fname_config[:-4] + '_' + datetime_str + '.txt'
system('cp ' + fname_config + ' ' + fname_config_new)

f_log = open('log_file_' + datetime_str + '.txt', 'w')
f_log.write('sim_num\tyear\tmonth\tpath2nprofiles\tfname_profile\tpath2hdf\tfname_hdf\tfname_config')
f_log.close()

#ii_nprof_start = 35*12 + 1
ii_nprof_start = 30*12 + 1
ii_nprof_end = ii_nprof_start + 12*10
#ii_nprof_end = nProfiles_all
for ii_nprof in range(ii_nprof_start, ii_nprof_end):
    print('run simulation for', year_l[ii_nprof], ' ', month_l[ii_nprof])
    fname_nprof = nprofile_list[ii_nprof]
    nprof_data, zprof_data = util.get_profile_from_file(fname_nprof)

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
    #The Script will dispatch jobs for the Frequency Range (Min to Max, i.e. 50 MHz to 450 MHz)
    freqMin = tx_signal_in.freqMin
    freqMax = tx_signal_in.freqMax
    ii_min = util.findNearest(freq_plus, freqMin)
    ii_max = util.findNearest(freq_plus, freqMax)
    # STEP 2 -> SEND OUT SCRIPTS

    print('any nans in spectrum?', np.isnan(np.any(tx_signal_in.spectrum_plus)))
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
            if ii_freq > 100:
                os.system('rm ' + fname_sh_in)
    fout_list.close()
    print('all jobs submitted',year_l[ii_nprof], ' ', month_l[ii_nprof])
    line_l = [str(ii_nprof),
              str(year_l[ii_nprof]),
              str(month_l[ii_nprof]),
              path2profiles,
              fname_nprof,
              dir_sim_path,
              fname_config_new]
    line_out = ''
    for k in range(len(line_l)):
        if k < len(line_l) -1:
            line_out += line_l[k] + '\t'
        else:
            line_out += line_l[k]
    line_out += '\n'
    f_log = open('log_file_' + datetime_str + '.txt', 'a')
    f_log.write(line_out)
    f_log.close()
    
    t_waiting = t_wait
    proceed_bool = False
    while proceed_bool == False:
        nJobs = countjobs()
        print('nJobs = ', nJobs)
        if t_waiting < 15*60:
            if nJobs > 0:
                print('Waiting for', t_wait, 's')
                time.sleep(t_wait)
                t_waiting += t_wait
            else:
                print('Jobs complete, proceed')
                proceed_bool = True
        else:
            print('Time out! Not all jobs terminatied after', t_wait*nLimit, 's')
            print('Abort, shut down all remaining jobs')
            system('./kill_jobs.sh')
            print('yiasou f**ken')
            exit()
    system('python add_spectrum_to_hdf.py ' + dir_sim_path + fname_list)
    system('python add_npy_to_hdf.py ' + dir_sim_path)
    print('Sim complete:', year_l[ii_nprof], ' ', month_l[ii_nprof], '\n')
    #TODO: Add a way to check that the simulation worked and produced valid results
