import time
import datetime
from sys import exit, argv

from os import system, popen
from ku_scripting import *

sys.path.append('../')

from data import create_sim, create_rxList_from_file, create_tx_signal, create_transmitter_array
from data import create_ascan_hdf, ascan
import util
from Askaryan_Signal import create_pulse as create_pulse_askaryan, TeV
import subprocess
from transmitter import tx_signal

'''
This script is designed from running a multi TX simualtion on a cluster - prevents too many .txt files from being generated at one time
'''

def countuserjobs():
    #Counts the number of jobs that is running under your username
    username = popen('whoami').read()[:-1]
    #print(username)
    cmd = r'squeue | grep "' + username + r'" | wc -l'
    #print(cmd)
    try:
        output = int(subprocess.check_output(cmd, shell=True))
    except:
        output = 0
    return output


def create_ascan_hdf2(fname_config, tx_signal, z_tx, nprof_data, zprof_data, fname_output):
    sim = create_sim(fname_config)
    rxList = create_rxList_from_file(fname_config)
    #tx_depths = create_transmitter_array(fname_config)
    tx_depths = [z_tx]

    output_hdf = h5py.File(fname_output, 'w')
    output_hdf.attrs["iceDepth"] = sim.iceDepth
    output_hdf.attrs["iceLength"] = sim.iceLength
    output_hdf.attrs["airHeight"] = sim.airHeight
    output_hdf.attrs["dx"] = sim.dx
    output_hdf.attrs["dz"] = sim.dz
    #output_hdf.attrs['n0'] = sim.n0

    output_hdf.attrs["Amplitude"] = tx_signal.amplitude
    output_hdf.attrs["freqCentral"] = tx_signal.frequency
    output_hdf.attrs["Bandwidth"] = tx_signal.bandwidth
    output_hdf.attrs["freqMax"] = tx_signal.freqMax
    output_hdf.attrs["freqMin"] = tx_signal.freqMin
    output_hdf.attrs["freqSample"] = tx_signal.fsample
    output_hdf.attrs["freqNyquist"] = tx_signal.freq_nyq
    output_hdf.attrs["tCentral"] = tx_signal.t_centre
    output_hdf.attrs["tSample"] = tx_signal.tmax
    output_hdf.attrs["dt"] = tx_signal.dt
    output_hdf.attrs["nSamples"] = tx_signal.nSamples

    output_hdf.create_dataset('n_profile', data=nprof_data)
    output_hdf.create_dataset('z_profile', data=zprof_data)
    output_hdf.create_dataset("source_depths", data=tx_depths)
    output_hdf.create_dataset('tspace', data=tx_signal.tspace)
    output_hdf.create_dataset('signalPulse', data=tx_signal.pulse)
    output_hdf.create_dataset('signalSpectrum', data=tx_signal.spectrum)

    rxList_positions = np.ones((len(rxList), 2))
    for i in range(len(rxList)):
        rx_i = rxList[i]
        rxList_positions[i, 0] = rx_i.x
        rxList_positions[i, 1] = rx_i.z

    output_hdf.create_dataset("rxList", data=rxList_positions)
    return output_hdf

def create_pulse(fname_config):
    config = configparser.ConfigParser()
    config.read(fname_config)
    config_tx_signal = config['TX_SIGNAL']
    tx_mode = config_tx_signal['mode']
    if tx_mode == 'gaussian':
        tx_signal_out = create_tx_signal(fname_config)
        tx_signal_out.get_gausspulse()
        tx_signal_out.add_gaussian_noise()
    elif tx_mode == 'impulse':
        #tx_signal_out = create_tx_signal(fname_config)
        fmin_BP = float(config_tx_signal['freqMin_BP'])
        fmax_BP = float(config_tx_signal['freqMax_BP'])
        t_centre = float(config_tx_signal['t_centre'])
        dt = float(config_tx_signal['dt'])
        band = float(fmax_BP - fmin_BP)
        t_max = float(config_tx_signal['t_max'])
        freq_centre = float(config_tx_signal['freq_centre'])
        
        tx_signal_out = tx_signal(frequency=freq_centre, bandwidth=band, t_centre=t_centre, dt=dt, tmax=t_max, amplitude=1)
        tx_signal_out.set_impulse()        
        tx_signal_out.apply_bandpass(fmin=fmin_BP, fmax=fmax_BP)
        #tx_signal_out.add_gaussian_noise()
    elif tx_mode == 'askaryan':
        ask_config = config['ASKARYAN']

        dtheta_v = float(ask_config['dtheta_v'])
        E_nu_eV = float(ask_config['showerEnergy'])
        E_nu_TeV = E_nu_eV / TeV
        R_alpha = float(ask_config['attenuationEquivalent'])
        tx_signal_out = create_tx_signal(fname_config)
        tspace = tx_signal_out.tspace
        tmax = tx_signal_out.tmax
        t_center = tx_signal_out.t_centre

        pulse_ask_out, tspace_ask = create_pulse_askaryan(Esh=E_nu_TeV, dtheta_v=dtheta_v, R_alpha=R_alpha,
                                                          t_min=-1 * t_center / 1e3, t_max=(tmax - t_center) / 1e3,
                                                          fs=tx_signal_out.fsample * 1e3)
        pulse_ask_interp = np.interp(tspace, tspace_ask, pulse_ask_out)
        tx_signal_out.set_pulse(pulse_ask_interp, tspace)
    return tx_signal_out

def create_spectrum(fname_config, nprof_data, zprof_data, fname_output_h5, z_tx=None):
    '''
    TO DO: change this to only have 1 tx
    Parameters
    ----------
    fname_config
    nprof_data
    zprof_data
    fname_output_h5

    Returns
    -------

    '''

    tx_signal_out = create_pulse(fname_config)
    if z_tx == None:
        hdf_ascan = create_ascan_hdf(fname_config=fname_config,
                                     tx_signal=tx_signal_out,
                                     nprof_data=nprof_data,
                                     zprof_data=zprof_data,
                                     fname_output=fname_output_h5)
    else:
        hdf_ascan = create_ascan_hdf2(fname_config=fname_config,
                                      z_tx=z_tx,tx_signal=tx_signal_out,
                                      nprof_data=nprof_data,
                                      zprof_data=zprof_data,
                                      fname_output=fname_output_h5)
    hdf_ascan.close()
    return tx_signal_out

nArgs = len(argv)
if nArgs == 2:
    fname_config = argv[1]
    fname_nprofile_in = None
elif nArgs == 3:
    fname_config = argv[1]
    fname_nprofile_in = argv[2]
    fname_base_name = os.path.basename(fname_nprofile_in)
    sim_prefix_in = fname_base_name.split('.')[0]
elif nArgs == 4:
    fname_config = argv[1]
    fname_nprofile_in = argv[2]
    sim_prefix_in = argv[3]
else:
    print('wrong arg number', nArgs)
    print('Enter: python ', argv[0], ' <config_file.txt>')
    exit()


config = configparser.ConfigParser()
config.read(fname_config)

if nArgs > 3:
    fname_nprof = config['REFRACTIVE_INDEX']['fname_profile']
else:
    fname_nprof = fname_nprofile_in
nprof_data, zprof_data = util.get_profile_from_file(fname_nprof)


dir_sim = config['OUTPUT']['path2output']
if os.path.isdir(dir_sim) == False:
    os.system('mkdir ' + dir_sim)
dir_sim_path = dir_sim + '/'

now = datetime.datetime.now()
datetime_str = now.strftime('%y%m%d_%H%M%S')
fname_config_new = fname_config[:-4] + '_' + datetime_str + '.txt'
#system('cp ' + fname_config + ' ' + fname_config_new)

if nArgs <= 2:
    sim_prefix = config['OUTPUT']['prefix'] #os.path.basename(fname_nprof)
else:
    sim_prefix = sim_prefix_in

#sim_prefix = sim_prefix[:-4]
sim_name = sim_prefix
fname_body = sim_prefix
fname_hdf0 = fname_body + '.h5'
fname_npy0 = fname_body + '.npy'

fname_hdf = dir_sim_path + fname_hdf0

tx_config = config['TRANSMITTER']
for key in tx_config.keys():
    print(key)
txList = []
if 'source_depth' in tx_config.keys():
    txList.append(float(tx_config['source_depth']))
elif 'fname_transmitters' in tx_config.keys():
    txList = create_transmitter_array(fname_config)
elif 'source_depth' in tx_config.keys() and 'fname_transmitters' in tx_config.keys():
    txList = create_transmitter_array(fname_config)
else:
    print('error, config[TRANSMITTER] in config file must have fname_transmitters or source_depth')
    exit()

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


t_wait_freq = 30
t_wait_tx = 60

nMinutes = 120
t_60 = 60.
max_wait_time = nMinutes * t_60
t_wait_it = 10.

fname_npy_l = []
fname_hdf_l = []
fname_txt_l = []
for ii_tx in range(nTx):
    z_tx = txList[ii_tx]
    print('z_tx = ', z_tx)

    fname_hdf_tx0 = fname_body + '_tx_' + str(round(z_tx)) + '.h5'
    fname_npy_tx0 = fname_body + '_tx_' + str(round(z_tx)) + '.npy'
    fname_hdf_tx = dir_sim_path + fname_hdf_tx0
    fname_npy_tx = dir_sim_path + fname_npy_tx0
    fname_npy_l.append(fname_npy_tx)
    fname_hdf_l.append(fname_hdf_tx)
    
    fname_tx_list = fname_body + '_tx_' + str(round(z_tx)) + '_list.txt'
    fname_txt_l.append(dir_sim_path + fname_tx_list)
    
    fout_list = open(dir_sim_path + fname_tx_list, 'w')
    fout_list.write(dir_sim_path + '\t' + fname_hdf_tx0 + '\t' + fname_npy_tx0 + '\n')
    fout_list.write(str(nTx) + '\t' + str(nRx) + '\t' + str(nSamples) + '\n')
    fout_list.write('ID_TX\tID_Freq\tFreq_GHz\tfname_npy\n')



    tx_signal_in_ii = create_spectrum(fname_config=fname_config,
                                   nprof_data=nprof_data, zprof_data=zprof_data,
                                   fname_output_h5=fname_hdf_tx, z_tx=z_tx)

    for ii_freq in range(ii_min, ii_max):
        jj_tx = ii_tx
        freq_ii = freq_plus[ii_freq]
        fname_txt_i = fname_body + '_tx_' + str(round(z_tx)) + '_' + str(jj_tx).zfill(2) + '_' + str(ii_freq) + '.txt'
        fname_txt_i = fname_txt_i
        fname_txt_path = os.path.join(dir_sim_path, fname_txt_i)
        print('create job for, z_tx = ', z_tx, ' m, f = ', freq_ii*1e3, ' MHz')

        line = str(jj_tx) + '\t' + str(ii_freq) + '\t' + str(round(freq_ii,3)) + '\t' + fname_txt_i + '\n'
        fout_list.write(line)

        cmd = 'python runSim_ascan_rx_from_txt.py ' + fname_config + ' '
        cmd += fname_txt_path + ' ' + fname_hdf_tx + ' ' + fname_nprof + ' '
        cmd += str(ii_freq) + ' ' + str(jj_tx)

        suffix = 'fid_' + str(jj_tx).zfill(2) + '_' + str(int(freq_ii*1e3))

        jobname = dir_sim_path + suffix
        fname_sh_in = 'sim_CFM_' + suffix + '.sh'

        fname_sh_out0 = 'sim_CFM_' + sim_name + '_' + suffix + '.out'
        fname_sh_out = dir_sim_path + fname_sh_out0

        make_job(fname_shell=fname_sh_in, fname_outfile=fname_sh_out, jobname=jobname, command=cmd)
        submit_job(fname_sh_in)
        os.system('rm ' + fname_sh_in)
        if ii_freq%100 == 0 and ii_freq > 0:
            print('wait ', t_wait_freq, 's, nJobs = ', countuserjobs())
            time.sleep(t_wait_freq)
    nJobs1 = countuserjobs()
    if nJobs1 > 0:
        proceed_bool = False
        t_waiting = t_wait_it
        while proceed_bool == False:
            nJobs2 = countuserjobs()
            print('nJobs =', nJobs2)
            if t_waiting < max_wait_time:
                if nJobs2 > 0:
                    print('Waiting for', t_wait_it, 's', ', total wait = ', t_waiting, 's')
                    time.sleep(t_wait_it)
                    t_waiting += t_wait_it
                else:
                    print('Jobs complete, proceed')
                    proceed_bool = True
            else:
                print('Time out! Not all jobs terminatied after', max_wait_time, 's')
                system('./kill-jobs.sh')
                print('Ending Sim')
                exit()
    fout_list.close()
    system('python add_spectrum_to_hdf.py ' + dir_sim_path + fname_tx_list)
    system('python add_npy_to_hdf.py ' + dir_sim_path)
    print('Simulation Complete for', z_tx)

# TODO: Last Step -> Combine ascans

ascan_all = ascan()
ascan_all.load_from_hdf(fname_hdf=fname_hdf)
nSamples = ascan_all.nSamples
fname_npy = dir_sim_path + fname_npy0
all_sim_npy = util.create_memmap(fname_npy, dimensions=(nTx, nRx, nSamples), data_type='complex')
print(all_sim_npy.shape)
for i in range(nTx):
    print(fname_npy_l[i], fname_npy)
    
    npy_i = np.load(fname_npy_l[i], 'r')
    print(npy_i, npy_i.shape)

    all_sim_npy[i,:,:] += npy_i[i,:,:]
ascan_all = ascan()
ascan_all.load_from_hdf(fname_hdf=fname_hdf)
ascan_all.save_spectrum(fname_npy=fname_npy)
print('All data saved to', fname_hdf, 'and', fname_npy)
print('deleting TX specific hdfs and npys')
for i in range(nTx):
    system('rm -f ' + fname_npy_l[i])
    system('rm -f ' + fname_hdf_l[i])
    system('rm -f ' + fname_txt_l[i])
print('Complete')
