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

def make_dB_plot(sim, fc, z_tx, label, fname_out):
    absu = np.transpose(abs(sim.get_field()))
    vmin = 1e-5
    vmax = 1e-2

    fig = pl.figure(figsize=(15, 5), dpi=120)

    #fig, axs = pl.subplots(1, 2, figsize=(10,5), dpi=120, gridspec_kw={'width_ratios': [1, 4]})
    #ax1 = axs[0]
    #ax2 = axs[1]
    ax2 = fig.add_subplot(111)
    #ax2 = fig.add_subplot(1,2,2)
    '''
    ax1.plot(sim.get_n(x=0), sim.get_z())
    ax1.legend()
    ax1.set_ylim(sim.iceDepth, -sim.airHeight)
    ax1.grid()
    ax1.axhline(z_tx,color='k')
    ax1.set_xlabel('Ref Index n')
    ax1.set_ylabel('Depth z [m]')
    ax1.set_xlim(1.3,1.8)
    '''
    ax2.set_title('Abs Field, ' + label + '\n$z_{tx} = $' + str(z_tx)
                 + ' m, f = $' + str(int(fc*1e3)) + ' \, \mathrm{MHz}$, $\lambda = $' +str(round(fc/0.3*1e3,2))+' cm')
    #pmesh = ax2.imshow(10*np.log10(absu), extent=[0, sim.x[-1], sim.z[-1], sim.z[0]],aspect='auto',cmap='viridis',vmin=vmin,vmax=vmax)
    pmesh = ax2.imshow(absu, extent=[0, sim.x[-1], sim.z[-1], sim.z[0]],
                      aspect='auto',cmap='viridis',vmin=vmin,vmax=vmax)
    cbar = pl.colorbar(pmesh, ax=ax2)
    #cbar.set_label('Amplitude [dB]')
    cbar.set_label('Amplitude [u]')
    #ax2.grid()
    ax2.scatter(0, z_tx,c='k')
    ax2.set_xlabel('Range X [m]')
    ax2.set_xlim(sim.dx, sim.iceLength)
    #ax2.set_ylabel('Depth Z [m]')
    if fname_out != None:
        fig.savefig(fname_out)
    pl.close(fig)

def compare_fields(sim, sim2, label0, label1, label2, fc, z_tx, mode ='ratio_dB', fname_out= None):
    field1 = abs(sim.get_field())
    field2 = abs(sim2.get_field())

    z1 = sim.get_z()
    n1 = sim.get_n(x=0)
    z2 = sim2.get_z()
    n2 = sim2.get_n(x=0)
    if mode == 'ratio_dB':
        absu = np.transpose(field2/field1)
        absu = 10*np.log10(absu)
        vmin = -30
        vmax = 0
    elif mode == 'relative_error':
        absu = np.transpose(abs(field2-field1)/abs(field1))
        vmin = 0
        vmax = 0.2
    fig, axs = pl.subplots(3, 1, figsize=(14,10), dpi=120, gridspec_kw={'height_ratios': [2, 1, 1]})
    ax0 = axs[1]
    ax1 = axs[2]
    ax2 = axs[0]
    if mode == 'relative_error':
        ax2.set_title('Relative Amplitude Offset $(\psi_{2} - \psi_{1})/ \psi_{1}$, ' + label0 + '\nf = $' + str(int(fc*1e3)) + ' \, \mathrm{MHz}$, $z_{tx} = $ ' + str(z_tx) + ' m')
    elif mode == 'ratio_dB':
        ax2.set_title('Amplitude Ratio $\psi_{2}/ \psi_{1}$, ' + label0 + '\nf = $' + str(int(fc*1e3)) + ' \, \mathrm{MHz}$, $z_{tx} = $ ' + str(z_tx) + ' m')

    pmesh = ax2.imshow(absu, extent=[0, sim.x[-1], sim.z[-1], sim.z[0]],
                      aspect='auto',cmap='coolwarm',vmin=vmin, vmax=vmax)
    cbar = pl.colorbar(pmesh, ax=ax2)
    if mode == 'ratio_dB':
        cbar.set_label('$10 \log{\psi_{2}/\psi_{1}}$ [dB]')
    elif mode == 'relative_error':
        cbar.set_label('$(\psi_{2} - \psi_{1})/ \psi_{1}$ [u]')
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
    pl.close(fig)

def histogram_ratio(sim1, sim2, label0, label1, label2, fc, z_tx, fname_out= None):
    field1 = 10*np.log10(abs(sim1.get_field()))
    field2 = 10*np.log10(abs(sim2.get_field()))
    gain = field2 - field1
    gain_h = []
    for i in range(len(gain)):
        for j in range(len(gain[0])):
            gain_h.append(gain[i,j])
    gain_bin_size = 0.5
    gain_min = -40 - gain_bin_size/2
    gain_max = 40 + gain_bin_size/2
    gain_width = gain_max - gain_min
    nBins = int(gain_width/gain_bin_size)

    fig = pl.figure(figsize=(8,5),dpi=120)
    ax = fig.add_subplot(111)
    ax.set_title('Abs Field, ' + label0 + ', ' + label1 + ' and ' + label2 + '\n$z_{tx} = $' + str(z_tx)
                 + ' m, f = $' + str(int(fc*1e3)) + ' \, \mathrm{MHz}$, $\lambda = $' +str(round(fc/0.3*1e3,2))+' cm')
    ax.hist(gain_h, range=(gain_min, gain_max), bins=nBins)
    ax.grid()
    ax.set_xlabel('Amp Gain [u]')
    ax.set_yscale('log')
    if fname_out != None:
        fig.savefig(fname_out)
    pl.close(fig)
fname_config = 'config_ICRC_summit.txt'
fname_CFM = 'CFMresults.hdf5'
fname_nprof = 'nProf_CFM.h5'
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


fname_list = []
n_matrix_yr = np.ones((nYears, nDepths))
for i in range(nYears):
    jj = util.findNearest(date_arr, year_list[i])
    year_id_list.append(jj)
    nprof_low_res_i = nprof_mat[jj]
    nprof_high_res_i = np.interp(z_prof_in, zprof_mat, nprof_low_res_i)
    n_matrix_yr[i] = nprof_high_res_i
n_prof_smooth = np.interp(z_prof_in, zprof_mat, nprof_mat[0])

z_list = [4, 8, 12, 16, 20, 40, 80]
nTx = len(z_list)
freq_list = np.arange(0.1, 0.5, 0.05)
nFreq = len(freq_list)
print(len(z_list), len(freq_list), nYears)
print('product:', nFreq*nTx*nYears, 'memory:', nFreq*nTx*nYears*292e3 / 1e6)
for i in range(nYears):
    path2plots0 = 'field_diff_plots_lin/'
    path2plots = path2plots0 + str(year_list[i])
    if os.path.isdir(path2plots) == False:
        os.system('mkdir ' + path2plots)
    path2plots += '/'
    path2plots_smooth = path2plots + 'smooth'
    if os.path.isdir(path2plots_smooth) == False:
        os.system('mkdir ' + path2plots_smooth)
    path2plots_smooth += '/'
    for j in range(nTx):
        for k in range(nFreq):
            print(year_list[i], z_list[j], freq_list[[k]])
            if i == 0:
                sim_smooth = solve_sim(fname_config, nprof_in=n_prof_smooth, zprof_in=z_prof_in,
                                       freq=freq_list[k], z_src=z_list[j])
                field_smooth = sim_smooth.get_field()

                sim0 = solve_sim(fname_config, nprof_in=n_matrix_yr[i], zprof_in=z_prof_in,
                             freq=freq_list[k], z_src=z_list[j])
                field0 = sim0.get_field()
                fname_out_dB =  path2plots + 'sim_yr_' + str(year_list[i]) + '_z=' + str(z_list[j]) + 'm'
                fname_out_dB += '_f=' + str(int(freq_list[k]*1e3)) + 'MHz'
                fname_out_dB += '_dB_plot.png'
                make_dB_plot(sim=sim0, fc=freq_list[k], z_tx=z_list[j],
                             label= 'Summit firn ' + str(year_list[i]),
                             fname_out=fname_out_dB)

                fname_out_dB2 = path2plots_smooth + 'sim_yr_' + str(year_list[i]) + '_z=' + str(z_list[j]) + 'm'
                fname_out_dB2 += '_f=' + str(int(freq_list[k] * 1e3)) + 'MHz'
                fname_out_dB2 += '_dB_plot.png'
                make_dB_plot(sim=sim_smooth, fc=freq_list[k], z_tx=z_list[j],
                             label='Summit firn ' + str(year_list[i]),
                             fname_out=fname_out_dB2)

                # RATIO -> Compare initial profile with year profile

                path2plots_smooth2 = path2plots_smooth + 'ratio'
                if os.path.isdir(path2plots_smooth2) == False:
                    os.system('mkdir ' + path2plots_smooth2)
                path2plots_smooth2 += '/'

                fname_out_ratio_smooth = path2plots_smooth2 + 'sim_yr_' + str(year_list[i]) + '_z=' + str(
                    z_list[j]) + 'm'
                fname_out_ratio_smooth += '_f=' + str(int(freq_list[k] * 1e3)) + 'MHz'
                fname_out_ratio_smooth += '_dB_plot.png'
                compare_fields(sim=sim_smooth, sim2=sim0, label0='Summit firn ',
                               label1='Smooth', label2=year_list[i], fc=freq_list[k],
                               z_tx=z_list[j], fname_out=fname_out_ratio_smooth)

                # HISTOGRAM -> Compare initial profile with year profile
                path2plots_smooth3 = path2plots_smooth + 'hist'
                if os.path.isdir(path2plots_smooth3) == False:
                    os.system('mkdir ' + path2plots_smooth3)
                path2plots_smooth3 += '/'

                fname_out_hist_smooth = path2plots_smooth3 + 'sim_yr_' + str(year_list[i]) + '_z=' + str(
                    z_list[j]) + 'm'
                fname_out_hist_smooth += '_f=' + str(int(freq_list[k] * 1e3)) + 'MHz'
                fname_out_hist_smooth += '_dB_plot.png'
                histogram_ratio(sim1=sim_smooth, sim2=sim0, label0='Summit firn ',
                                label1='Smooth', label2=str(year_list[i]), fc=freq_list[k],
                                z_tx=z_list[j], fname_out=fname_out_hist_smooth)
            else:
                sim_smooth = solve_sim(fname_config, nprof_in=n_prof_smooth, zprof_in=z_prof_in,
                                       freq=freq_list[k], z_src=z_list[j])
                sim0 = solve_sim(fname_config, nprof_in=n_matrix_yr[i - 1], zprof_in=z_prof_in,
                                 freq=freq_list[k], z_src=z_list[j])
                field0 = sim0.get_field()
                sim1 = solve_sim(fname_config, nprof_in=n_matrix_yr[i], zprof_in=z_prof_in,
                                 freq=freq_list[k], z_src=z_list[j])
                field1 = sim1.get_field()
                fname_out_dB =  path2plots + 'sim_yr_' + str(year_list[i]) + '_z=' + str(z_list[j]) + 'm'
                fname_out_dB += '_f=' + str(int(freq_list[k]*1e3)) + 'MHz'
                fname_out_dB += '_dB_plot.png'
                make_dB_plot(sim=sim1, fc=freq_list[k], z_tx=z_list[j],
                             label= 'Summit firn ' + str(year_list[i]),
                             fname_out=fname_out_dB)

                path2plots2 = path2plots + 'ratio'
                if os.path.isdir(path2plots2) == False:
                    os.system('mkdir ' + path2plots2)
                path2plots2 += '/'

                path2plots3 = path2plots + 'ratio_hist'
                if os.path.isdir(path2plots3) == False:
                    os.system('mkdir ' + path2plots3)
                path2plots3 += '/'

                #RATIO -> Compare initial profile with year profile
                path2plots_smooth2 = path2plots_smooth + 'ratio'
                if os.path.isdir(path2plots_smooth2) == False:
                    os.system('mkdir ' + path2plots_smooth2)
                path2plots_smooth2 += '/'
                fname_out_ratio_smooth = path2plots_smooth2 + 'sim_yr_' + str(year_list[i]) + '_z=' + str(z_list[j]) + 'm'
                fname_out_ratio_smooth += '_f=' + str(int(freq_list[k] * 1e3)) + 'MHz'
                fname_out_ratio_smooth += '_dB_plot.png'
                compare_fields(sim=sim_smooth, sim2=sim1, label0='Summit firn ',
                               label1='Smooth',label2=year_list[i], fc=freq_list[k],
                               z_tx=z_list[j], fname_out=fname_out_ratio_smooth)


                #RATIO: year vs year
                fname_out_ratio = path2plots2 + 'sim_yr_' + str(year_list[i]) + '_z=' + str(z_list[j]) + 'm'
                fname_out_ratio += '_f=' + str(int(freq_list[k] * 1e3)) + 'MHz'
                fname_out_ratio += '_ratio_plot.png'
                compare_fields(sim=sim0, sim2=sim1, label0='Summit firn ',
                               label1=year_list[i - 1], label2=year_list[i], fc=freq_list[k],
                               z_tx=z_list[j], mode='ratio_dB', fname_out=fname_out_ratio)
                fname_out_ratio = path2plots2 + 'sim_yr_' + str(year_list[i]) + '_z=' + str(z_list[j]) + 'm'
                fname_out_ratio += '_f=' + str(int(freq_list[k] * 1e3)) + 'MHz'
                fname_out_ratio += '_rel_error_plot.png'
                compare_fields(sim=sim0, sim2=sim1, label0='Summit firn ',
                               label1=year_list[i-1],label2=year_list[i], fc=freq_list[k],
                               z_tx=z_list[j],mode='relative_error', fname_out=fname_out_ratio)

                #HISTOGRAM
                fname_out_hist = path2plots3 + 'sim_yr_' + str(year_list[i]) + '_z=' + str(z_list[j]) + 'm'
                fname_out_hist += '_f=' + str(int(freq_list[k] * 1e3)) + 'MHz'
                fname_out_hist += '_ratio_histogram.png'
                histogram_ratio(sim1=sim0, sim2=sim1, label0='Summit firn ',
                                label1=str(year_list[i-1]),label2=str(year_list[i]), fc=freq_list[k],
                                z_tx=z_list[j], fname_out=fname_out_hist)

                #HISTOGRAM -> Compare initial profile with year profile
                path2plots_smooth3 = path2plots_smooth + 'hist'
                if os.path.isdir(path2plots_smooth3) == False:
                    os.system('mkdir ' + path2plots_smooth3)
                path2plots_smooth3 += '/'

                fname_out_hist_smooth = path2plots_smooth3 + 'sim_yr_' + str(year_list[i]) + '_z=' + str(
                    z_list[j]) + 'm'
                fname_out_hist_smooth += '_f=' + str(int(freq_list[k] * 1e3)) + 'MHz'
                fname_out_hist_smooth += '_dB_plot.png'
                histogram_ratio(sim1=sim_smooth, sim2=sim1, label0='Summit firn ',
                                label1='Smooth', label2=str(year_list[i]), fc=freq_list[k],
                                z_tx=z_list[j], fname_out=fname_out_hist_smooth)

                '''
                for m in range(len(field0)):
                    for n in range(len(field0[0])):
                        field_diff_r = field1.real - field0.real
                        field_diff_r_scale = field_diff_r * field0
                        field_rati0 = field1/field0
                '''

