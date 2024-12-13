from sys import exit, argv
import time
import datetime
from os import system
import numpy as np
from numpy.fft import fft, rfft, fftfreq, rfftfreq, ifft, fftshift, ifftshift

from ku_scripting import *

sys.path.append('../')
from data import create_sim, create_rxList_from_file, create_tx_signal, create_transmitter_array
from data import create_ascan_hdf, ascan
import util
from data import create_transmitter_array_from_file

'''
This Script is utilized for a study on how changes in firn density affects 
radio propagation from a deep RF signal
'''

# Ingredients
# TX Pulse - FNAME INPUT
# Ref Index Profile - FNAME INPUT
# Geometry - FNAME INPUT
# RX Positions - FNAME INPUT
# TX Positions - FNAME INPUT

if len(argv) == 2:
    fname_config = sys.argv[1]
else:
    print('wrong arg number', len(argv), 'should be ', 2)
    print('Enter: python ', argv[0], ' <config_file.txt>')
    exit()

t_wait = 10 # Seconds
t_minute = 60 # Seconds
nMinutes = 15 # Number of Minutes to Wait

config = configparser.ConfigParser()
config.read(fname_config)
dir_sim = config['OUTPUT']['path2output']
if os.path.isdir(dir_sim) == False:
    os.system('mkdir ' + dir_sim)
dir_sim_path = dir_sim + '/'

#Filename for the Ref Index Profile
fname_nprof = config['REFRACTIVE_INDEX']['fname_profile']
nprof_data, zprof_data = util.get_profile_from_file(fname_nprof)

#Filename for the list of Transmitters
#fname_transmitters = config['TRANSMITTER']['fname_transmitters']
txList = create_transmitter_array_from_file(fname_config)
nTx = len(txList)

#TX Signal
tx_signal_in = create_tx_signal(fname_config)
tx_signal_in.set_impulse()

#tx_signal_in.apply_lowpass(0.2)
tx_signal_in.apply_bandpass(0.05, 0.35)
tspace = tx_signal_in.tspace

#RxList
rxList = create_rxList_from_file(fname_config)
nRx = len(rxList)

now = datetime.datetime.now()
datetime_str = now.strftime('%y%m%d_%H%M%S')

#fname_config_new = fname_config[:-4] + '_' + datetime_str + '.txt'
#system('cp ' + fname_config + ' ' + fname_config_new)
sim_prefix = config['OUTPUT']['prefix']
sim_name = sim_prefix

fname_body = os.path.basename(fname_nprof)
fname_body = fname_body[:-4]
fname_hdf0 = fname_body + '.h5'
fname_npy0 = fname_body + '.npy'
fname_hdf = dir_sim_path + fname_hdf0

tx_pulse_in = tx_signal_in.pulse
tx_spectrum_in = tx_signal_in.spectrum_plus
freq_plus = tx_signal_in.freq_plus
tspace = tx_signal_in.tspace
nSamples = tx_signal_in.nSamples

fig = pl.figure(figsize=(8,5),dpi=120)
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)
ax1.plot(tspace, tx_signal_in.pulse.real)
ax2.plot(freq_plus, abs(tx_spectrum_in))
ax1.grid()
ax2.grid()
pl.show()

print(fname_hdf)

hdf_ascan = create_ascan_hdf(fname_config=fname_config,
                             tx_signal=tx_signal_in,
                             nprof_data=nprof_data,
                             zprof_data=zprof_data,
                             fname_output=fname_hdf)
hdf_ascan.close()


#TODO: Distinguish between freqMin and freqMax (minimum and maximum frequency of simulation scan) & the filter frequencies
freqMin = tx_signal_in.freqMin
freqMax = tx_signal_in.freqMax
ii_min = util.findNearest(freq_plus, freqMin)
ii_max = util.findNearest(freq_plus, freqMax)
#TODO: Finish Writing File ->
print('any nans in spectrum?', np.isnan(np.any(tx_signal_in.spectrum_plus)))

#Write the Name of the File
fname_list = fname_body + '_list.txt'

fout_list = open(dir_sim_path + fname_list, 'w')
fout_list.write(dir_sim_path+ '\t' + fname_hdf0 + '\t' + fname_npy0 +'\n')
fout_list.write(str(nTx) + '\t' + str(nRx) + '\t' + str(nSamples) + '\n')
fout_list.write('ID_TX\tID_Freq\tFreq_GHz\tfname_npy\n')

#ii_max2 = ii_min + 1

for ii_tx in range(nTx):
    z_tx = txList[ii_tx]

    for ii_freq in range(ii_min, ii_max):
        freq_ii = freq_plus[ii_freq]

        fname_txt_i = fname_body + '_' + str(ii_tx).zfill(2) + '_' + str(ii_freq) + '.txt'
        fname_txt_i = fname_txt_i
        fname_txt_path = os.path.join(dir_sim_path, fname_txt_i)
        print('create job for, z_tx = ', z_tx, ' m, f = ', freq_ii * 1e3, ' MHz')

        line = str(ii_tx) + '\t' + str(ii_freq) + '\t' + str(round(freq_ii, 3)) + '\t' + fname_txt_i + '\n'

        cmd = 'python runSim_ascan_rx_from_txt.py ' + fname_config + ' '
        cmd += fname_txt_path + ' ' + fname_hdf + ' ' + fname_nprof + ' '
        cmd += str(ii_freq) + ' ' + str(ii_tx)
        #system(cmd)
        fout_list.write(line)
        
        suffix = 'fid_' + str(ii_tx).zfill(2) + '_' + str(int(freq_ii * 1e3))

        jobname = dir_sim_path + suffix
        fname_sh_in = 'sim_CFM_' + suffix + '.sh'

        fname_sh_out0 = 'sim_CFM_' + sim_name + '_' + suffix + '.out'
        fname_sh_out = dir_sim_path + fname_sh_out0

        make_job(fname_shell=fname_sh_in, fname_outfile=fname_sh_out, jobname=jobname, command=cmd)
        submit_job(fname_sh_in)
        if ii_freq > 100:
            os.system('rm ' + fname_sh_in)



    t_waiting = t_wait
    proceed_bool = False
    while proceed_bool == False:
        nJobs = countjobs()
        print('nJobs = ', nJobs)
        if t_waiting < nMinutes * t_minute:
            if nJobs > 0:
                print('Waiting for', t_wait, 's')
                time.sleep(t_wait)
                t_waiting += t_wait
            else:
                print('Jobs complete, proceed')
                proceed_bool = True
    else:
        print('Time out! Not all jobs terminatied after', nMinutes*t_minute, 's')
        print('Abort, shut down all remaining jobs')
        system('./kill_jobs.sh')
        exit()

system('python add_spectrum_to_hdf.py ' + dir_sim_path + fname_list)
system('python add_npy_to_hdf.py ' + dir_sim_path)
print('Sim complete')

fout_list.close()
