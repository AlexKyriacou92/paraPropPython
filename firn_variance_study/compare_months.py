import os.path
import sys
import numpy as np
import time
import datetime
import h5py
import configparser
import matplotlib.pyplot as pl

sys.path.append('../')

import util
import paraPropPython as ppp
from receiver import receiver as rx
from transmitter import tx_signal
from data import create_sim, create_rxList_from_file, create_transmitter_array, create_hdf_FT
from data import create_tx_signal, bscan, bscan_rxList, create_hdf_bscan
def solve_sim(fname_config_in, nprof_in, zprof_in, freq, z_src):
    sim = create_sim(fname_config_in)
    sim.set_n(nVec=nprof_in, zVec=zprof_in)
    sim.set_dipole_source_profile(centerFreq=freq, depth=z_src)
    sim.set_cw_source_signal(freq)
    sim.do_solver()
    return sim


def get_data(fname_in, input_var='density', jj = 0):
    with  h5py.File(fname_in, 'r') as input_hdf:
        z_arr = np.array(input_hdf['depth'])
        data_matrix = np.array(input_hdf[input_var])
    data_arr = data_matrix[jj]
    data_arr = data_arr[1:]
    z_arr = z_arr[1:]
    return data_arr, z_arr

def get_data_matrix(fname_in, input_var='density'):
    with  h5py.File(fname_in, 'r') as input_hdf:
        z_arr = np.array(input_hdf['depth'])
        data_matrix = np.array(input_hdf[input_var])
    data_matrix = data_matrix[:,1:]
    z_arr = z_arr[1:]
    return data_matrix, z_arr

def get_dates(fname_in):
    with  h5py.File(fname_in, 'r') as input_hdf:
        data_matrix = np.array(input_hdf['density'])
    date_arr = data_matrix[1:,0]
    return date_arr

def make_dB_plot(sim, fc, z_tx, label, vmin=-80, vmax=-10, mode = 'dB', show_bool=False, fname_out=None):
    absu = np.transpose(abs(sim.get_field()))
    fig = pl.figure(figsize=(15, 5), dpi=120)
    ax2 = fig.add_subplot(111)

    ax2.set_title('Abs Field, ' + label + '\n$z_{tx} = $' + str(z_tx)
                 + ' m, f = $' + str(int(fc*1e3)) + ' \, \mathrm{MHz}$, $\lambda = $' +str(round(fc/0.3*1e3,2))+' cm')
    if mode == 'dB':
        pmesh = ax2.imshow(10*np.log10(absu), extent=[0, sim.x[-1], sim.z[-1], sim.z[0]],
                           aspect='auto',cmap='viridis',vmin=vmin,vmax=vmax)
        cbar = pl.colorbar(pmesh, ax=ax2)
        cbar.set_label('Amplitude [dB]')
    elif mode == 'linear':
        pmesh = ax2.imshow(absu, extent=[0, sim.x[-1], sim.z[-1], sim.z[0]],
                      aspect='auto',cmap='viridis',vmin=vmin,vmax=vmax)
        cbar = pl.colorbar(pmesh, ax=ax2)
        cbar.set_label('Amplitude [u]')
    ax2.scatter(0, z_tx,c='k')
    ax2.set_xlabel('Range X [m]')
    ax2.set_xlim(sim.dx, sim.iceLength)
    ax2.set_ylabel('Depth Z [m]')
    if fname_out != None:
        fig.savefig(fname_out)
    if show_bool == True:
        pl.show()
    else:
        pl.close(fig)

def compare_fields(sim, sim2, label0, label1, label2, fc, z_tx,show_bool=False, mode ='ratio_dB', fname_out= None):
    field1 = abs(sim.get_field())
    field2 = abs(sim2.get_field())

    z1 = sim.get_z()
    n1 = sim.get_n(x=0)
    z2 = sim2.get_z()
    n2 = sim2.get_n(x=0)
    if mode == 'ratio_dB':
        absu = np.transpose(field2/field1)
        absu = 10*np.log10(absu)
        vmin = -20
        vmax = 20
    elif mode == 'relative_error':
        absu = np.transpose(abs(field2-field1)/abs(field1))
        vmin = 0
        vmax = 0.2
    elif mode == 'abs_error':
        absu = np.transpose(abs(field2-field1))
        absu = 10*np.log10(absu)
        vmax = -10
        vmin = -80
    fig, axs = pl.subplots(3, 1, figsize=(14,10), dpi=120, gridspec_kw={'height_ratios': [2, 1, 1]})
    ax0 = axs[1]
    ax1 = axs[2]
    ax2 = axs[0]
    if mode == 'relative_error':
        ax2.set_title('Relative Amplitude Offset $(\psi_{2} - \psi_{1})/ \psi_{1}$, ' + label0 + '\nf = $' + str(int(fc*1e3)) + ' \, \mathrm{MHz}$, $z_{tx} = $ ' + str(z_tx) + ' m')
    elif mode == 'ratio_dB':
        ax2.set_title('Amplitude Ratio $\psi_{2}/ \psi_{1}$, ' + label0 + '\nf = $' + str(int(fc*1e3)) + ' \, \mathrm{MHz}$, $z_{tx} = $ ' + str(z_tx) + ' m')
    elif mode == 'abs_error':
        ax2.set_title('Amplitude Offset $|\psi_{2} - \psi_{1}|$, ' + label0 + '\nf = $' + str(int(fc*1e3)) + ' \, \mathrm{MHz}$, $z_{tx} = $ ' + str(z_tx) + ' m')


    if mode == 'abs_error':
        pmesh = ax2.imshow(absu, extent=[0, sim.x[-1], sim.z[-1], sim.z[0]],
                      aspect='auto',cmap='coolwarm',vmin=vmin, vmax=vmax)
    else:
        pmesh = ax2.imshow(absu, extent=[0, sim.x[-1], sim.z[-1], sim.z[0]],
                      aspect='auto',cmap='coolwarm',vmin=vmin, vmax=vmax)
    cbar = pl.colorbar(pmesh, ax=ax2)
    if mode == 'ratio_dB':
        cbar.set_label('$10 \log{\psi_{2}/\psi_{1}}$ [dB]')
    elif mode == 'relative_error':
        cbar.set_label('$(\psi_{2} - \psi_{1})/ \psi_{1}$ [u]')
    elif mode == 'abs_error':
        cbar.set_label('$|\psi_{2} - \psi_{1}|$ [u]')

    ax2.scatter(0, z_tx, c='k')
    ax2.set_xlabel('Range X [m]')
    ax2.set_xlim(sim.dx, sim.iceLength)
    ax2.set_ylabel('Depth Z [m]')

    ax1.plot(z1, n2-n1,c='k')
    ax1.legend()
    ax1.set_xlim(-sim.airHeight, sim.iceDepth)
    ax1.grid()
    ax1.set_ylabel('Ref Index Offset $\Delta n = n_{1} - n_{1}$')
    ax1.set_xlabel('Depth z [m]')

    ax0.plot(z1, n1, label=label1)
    ax0.plot(z2, n2, label=label2)
    ax0.axvline(z_tx,color='k')
    ax0.legend()
    ax0.set_xlim(-sim.airHeight, sim.iceDepth)
    ax0.grid()
    ax0.set_ylabel('Ref Index n')
    ax0.set_xlabel('Depth z [m]')
    ax0.set_ylim(1.3,1.8)
    if fname_out != None:
        fig.savefig(fname_out)
    if show_bool == False:
        pl.close(fig)
    else:
        pl.show()

def histogram_ratio(sim1, sim2, label0, label1, label2, fc, z_tx, bin_min=-40, bin_max=40, dbin = 0.5, mode='ratio_dB', show_bool=False, fname_out= None):
    if mode == 'ratio_dB':
        field1 = 10*np.log10(abs(sim1.get_field()))
        field2 = 10*np.log10(abs(sim2.get_field()))
        gain = field2 - field1
    elif mode == 'relative_error':
        field1 = abs(sim1.get_field())
        field2 = abs(sim2.get_field())
        gain = abs(field2-field1)/field1
    elif mode == 'abs_error':
        field1 = abs(sim1.get_field())
        field2 = abs(sim2.get_field())
        gain = 10*np.log10(abs(field2-field1))
    gain_h = []
    for i in range(len(gain)):
        for j in range(len(gain[0])):
            gain_h.append(gain[i,j])
    gain_bin_size = dbin
    gain_min = bin_min - gain_bin_size/2
    gain_max = bin_max + gain_bin_size/2
    gain_width = gain_max - gain_min
    nBins = int(gain_width/gain_bin_size)

    fig = pl.figure(figsize=(8,5),dpi=120)
    ax = fig.add_subplot(111)
    ax.set_title('Abs Field, ' + label0 + ', ' + label1 + ' and ' + label2 + '\n$z_{tx} = $' + str(z_tx)
                 + ' m, f = $' + str(int(fc*1e3)) + ' \, \mathrm{MHz}$, $\lambda = $' +str(round(fc/0.3*1e3,2))+' cm')
    ax.hist(gain_h, range=(gain_min, gain_max), bins=nBins)
    ax.grid()
    if mode == 'ratio_dB':
        ax.set_xlabel('Amp Gain [dB]')
    elif mode == 'relative_error':
        ax.set_xlabel('Relative Error [u]')
    ax.set_yscale('log')
    if fname_out != None:
        fig.savefig(fname_out)
    if show_bool == False:
        pl.close(fig)
    else:
        pl.show()

fname_config = 'config_ICRC_summit_km.txt'
fname_CFM = 'CFMresults.hdf5'
fname_nprof = 'nProf_CFM_deep.h5'
date_arr = get_dates(fname_CFM)
nprofile_hdf = h5py.File(fname_nprof, 'r')
nprof_mat = np.array(nprofile_hdf.get('n_profile_matrix'))
zprof_mat = np.array(nprofile_hdf.get('z_profile'))
nprofile_hdf.close()
sim0 = create_sim(fname_config)
dz = sim0.dz
z_prof_in = np.arange(min(zprof_mat), sim0.iceDepth, dz)
nDepths = len(z_prof_in)


#start_year = int(date_arr[0])
start_year = 2011
end_year = int(date_arr[-1]) + 1
year_list = np.arange(start_year, end_year, 1)
year_id_list = []
print(year_list)
nYears = len(year_list)

nDates = len(date_arr)
fname_list = []
n_matrix_yr = np.ones((nDates, nDepths))
for i in range(nDates):
    nprof_low_res_i = nprof_mat[i]
    nprof_high_res_i = np.interp(z_prof_in, zprof_mat, nprof_low_res_i)
    n_matrix_yr[i] = nprof_high_res_i
n_prof_smooth = np.interp(z_prof_in, zprof_mat, nprof_mat[0])

freq = float(sys.argv[1])
z_tx = float(sys.argv[2])
year1 = int(sys.argv[3])
month1 = int(sys.argv[4])
year2 = int(sys.argv[5])
month2 = int(sys.argv[6])

date_1 = float(year1) + float(month1)/12.
date_2 = float(year2) + float(month2)/12.

ii_1 = util.findNearest(date_arr, date_1)
ii_2 = util.findNearest(date_arr, date_2)

nprof_1 = n_matrix_yr[ii_1]
nprof_2 = n_matrix_yr[ii_2]
sim1 = solve_sim(fname_config_in=fname_config, nprof_in=nprof_1, zprof_in=z_prof_in, freq=freq, z_src=z_tx)
sim2 = solve_sim(fname_config_in=fname_config, nprof_in=nprof_2, zprof_in=z_prof_in, freq=freq, z_src=z_tx)

label1 = str(year1) + ' ' + str(month1)
label2 = str(year2) + ' ' + str(month2)
ratio_label = 'Ratio of amplitude [dB] '
rel_err_label = 'Relative Error [u] '

path2plots = 'plots2/'
mid_path = '_f=' + str(int(freq*1e3)) + 'MHz_Ztx=' + str(round(z_tx,2)) + 'm_'
amp_prefix = path2plots + 'amplitude' + mid_path

show_bool0 = False
make_dB_plot(sim1, fc=freq, z_tx=z_tx, label=label1, vmin=-40, vmax=-10,
             mode = 'dB', show_bool=show_bool0,
             fname_out=amp_prefix+label1+'_lin.png')
make_dB_plot(sim1, fc=freq, z_tx=z_tx, label=label1, vmin=1e-4, vmax=1e-1,
             mode = 'linear', show_bool=show_bool0,
             fname_out=amp_prefix+label1+'_dB.png')
make_dB_plot(sim2, fc=freq, z_tx=z_tx, label=label2, vmin=-40, vmax=-10,
             mode = 'dB', show_bool=show_bool0,
             fname_out=amp_prefix+label2+'_lin.png')
make_dB_plot(sim2, fc=freq, z_tx=z_tx, label=label2, vmin=1e-4, vmax=1e-1,
             mode = 'linear', show_bool=show_bool0,
             fname_out=amp_prefix+label2+'_dB.png')

abs_err_prefix = path2plots + 'abs_err' + mid_path
compare_fields(sim=sim1, sim2=sim2,
               label0= rel_err_label, label1=label1, label2=label2,
               fc=freq, z_tx=z_tx,
               show_bool=show_bool0, mode='abs_error',
               fname_out=abs_err_prefix + label1 + '_' + label2 + '_field.png')
histogram_ratio(sim1=sim1, sim2=sim2,
                label0=ratio_label, label1=label1, label2=label2,
                fc=freq, z_tx=z_tx, bin_min=-90, bin_max=0, dbin=1,
                show_bool=show_bool0, mode='abs_error',
                fname_out=abs_err_prefix + label1 + '_' + label2 + '_hist.png')

rel_err_prefix = path2plots + 'rel_err' + mid_path
compare_fields(sim=sim1, sim2=sim2,
               label0= rel_err_label, label1=label1, label2=label2,
               fc=freq, z_tx=z_tx,
               show_bool=show_bool0, mode='relative_error',
               fname_out=rel_err_prefix + label1 + '_' + label2 + '_field.png')
histogram_ratio(sim1=sim1, sim2=sim2,
                label0=ratio_label, label1=label1, label2=label2,
                fc=freq, z_tx=z_tx, bin_min=0, bin_max=20, dbin=0.1,
                show_bool=show_bool0, mode='relative_error', fname_out=rel_err_prefix + label1 + '_' + label2 + '_hist.png')

ratio_prefix = path2plots + 'ratio' + mid_path
compare_fields(sim=sim1, sim2=sim2,
               label0= rel_err_label, label1=label1, label2=label2,
               fc=freq, z_tx=z_tx,
               show_bool=show_bool0, mode='ratio_dB',
               fname_out=ratio_prefix + label1 + '_' + label2 + '_field.png')
histogram_ratio(sim1=sim1, sim2=sim2,
                label0=ratio_label, label1=label1, label2=label2,
                fc=freq, z_tx=z_tx, bin_min=-30, bin_max=30, dbin=0.1,
                show_bool=show_bool0, mode='ratio_dB',
                fname_out=ratio_prefix + label1 + '_' + label2 + '_hist.png')
