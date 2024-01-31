import os.path
import sys
import numpy as np
import time
import h5py

sys.path.append('../')
from data import ascan
import util

fname_list = sys.argv[1]

fin_list = open(fname_list, 'r')
cols0 = fin_list.readline().split()
dir_sim_path = cols0[0]
fname_hdf0 = cols0[1]
fname_npy0 = cols0[2]

fname_hdf = dir_sim_path + fname_hdf0
fname_npy = dir_sim_path + fname_npy0
print(fname_npy)
cols1 = fin_list.readline().split()
nTx = int(cols1[0])
nRx = int(cols1[1])
nSamples = int(cols1[2])

cols2 = fin_list.readline()

spectrum = util.create_memmap(fname_npy, dimensions=(nTx, nRx, nSamples), data_type='complex')

for line in fin_list:
    cols_l = line.split()
    ii_tx = int(cols_l[0])
    ii_freq = int(cols_l[1])
    freq_ii = float(cols_l[2])
    fname_spectrum_ii = dir_sim_path + cols_l[3]

    if fname_spectrum_ii[-3:] == 'npy':
        spectrum_i = np.load(fname_spectrum_ii, 'r')
        spectrum[:, :, ii_freq] = spectrum_i
    elif fname_spectrum_ii[-3:] == 'txt':
        spectrum_i = np.zeros((nTx, nRx),dtype='complex')

        with open(fname_spectrum_ii, 'r') as fin:
            for _ in range(2):
                next(fin)
            jj_rx = 0
            for line in fin:
                cols = line.split()
                rx_x = float(cols[0])
                rx_z = float(cols[1])
                amp_rx_re = float(cols[2])
                amp_rx_im = float(cols[3])
                amp_rx = amp_rx_re + 1j*amp_rx_im

                spectrum_i[ii_tx,jj_rx] = amp_rx
                jj_rx += 1
            spectrum[:, :, ii_freq] = spectrum_i

'''
print(fname_hdf)
with h5py.File(fname_hdf, 'a') as hdf_in:
    hdf_in.create_dataset('rxSpectrum', data=spectrum)
    pulse_arr = np.zeros((nTx, nRx, nSamples),dtype='complex')
    for i in range(nTx):
        for j in range(nRx):
            spectrum_ij = spectrum[i,j]
            spectrum_ij_flip = np.flip(spectrum_ij)
            pulse_rx = np.fft.ifft(spectrum_ij_flip)
            pulse_rx[i,j] = pulse_rx
    hdf_in.create_dataset('rxSignal', data=pulse_arr)
'''
#TODO: DELETE INDIVIDUAL NPY FILES
#TODO: DELETE OUT FILES

'''
with open(fname_list) as fin_list2:
    for _ in range(3):
        next(fin_list2)
    for line in fin_list2:
        cols_l = line.split()
        cols_l = line.split()
        ii_freq = int(cols_l[0])
        freq_ii = float(cols_l[1])
        fname_spectrum_ii = dir_sim_path + cols_l[2]

        os.system('rm -f ' + fname_spectrum_ii)
os.system('rm -f ' + dir_sim_path + '.out')
'''