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

t_wait = 10 # 10 Minutes

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
tx_signal_in.apply_bandpass(tx_signal_in.df, 0.3)
tspace = tx_signal_in.tspace

#RxList
rxList = create_rxList_from_file(fname_config)
nRx = len(rxList)

'''
fname_tx_signal = config['TX_SIGNAL']['fname_tx_signal']
ampTx_data, t_data = util.get_profile_from_file(fname_tx_signal)
print(t_data, ampTx_data)

freq = fftfreq(len(t_data), t_data[1]-t_data[0])
ampTx_spec = fft(ampTx_data)
ampTx_spec = util.hilbertEnvelope(ampTx_spec.real)



fig = pl.figure(figsize=(8,5),dpi=120)
ax = fig.add_subplot(111)
ax2 = ax.twinx()
ax.plot(freq, ampTx_spec,c='k')
ax2.plot(tx_signal_in.freq_space, abs(tx_signal_in.spectrum),c='m')
ax.grid()
pl.show()

fig = pl.figure(figsize=(8,5),dpi=120)
ax1 = fig.add_subplot(111)
ax2 = ax1.twinx()
ax1.plot(tspace, util.hilbertEnvelope(tx_signal_in.pulse.real),c='m',label='Simulated')
ax2.plot(t_data, ampTx_data,c='k',label='Data')
#ax2.plot(t_data, ifft(abs(ampTx_spec)))
ax1.grid()
ax1.set_xlim(-10,15)
ax2.set_xlim(-10,15)
pl.show()

fig = pl.figure(figsize=(8,5),dpi=120)
ax = fig.add_subplot(111)
ax.plot(t_data, ampTx_data)
ax.grid()
pl.show()
'''

now = datetime.datetime.now()
datetime_str = now.strftime('%y%m%d_%H%M%S')

#fname_config_new = fname_config[:-4] + '_' + datetime_str + '.txt'
#system('cp ' + fname_config + ' ' + fname_config_new)
sim_prefix = config['INPUT']['prefix']
fname_body = os.path.basename(fname_nprof)
fname_body = fname_body[:-4]
fname_hdf0 = fname_body + '.h5'
fname_npy0 = fname_body + '.npy'
fname_hdf = dir_sim_path + fname_hdf0


hdf_ascan = create_ascan_hdf(fname_config=fname_config,
                             tx_signal=tx_signal_in,
                             nprof_data=nprof_data,
                             zprof_data=zprof_data,
                             fname_output=fname_hdf)

tx_pulse_in = tx_signal_in.pulse
tx_spectrum_in = tx_signal_in.spectrum_plus

freq_plus = tx_signal_in.freq_plus
tspace = tx_signal_in.tspace
nSamples = tx_signal_in.nSamples

#TODO: Finish Writing File ->
'''
#It should have the ability to:
-Simulate for 3000 m x 1500 m for a given input ref index
-
'''
