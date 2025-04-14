import time
import datetime
import h5py
import numpy as np

from sys import argv
from os.path import join, isfile

from NuRadioMC.SignalProp import propagation
from NuRadioMC.SignalProp.analyticraytracing import solution_types, ray_tracing_2D
from NuRadioMC.utilities import medium

import sys
sys.path.append('../')
import util
from data import ascan

def get_ray_times(x_tx, z_tx, x_rx, z_rx, mode='radiopropa'):
    if mode != 'radiopropa' and mode != 'analytic':
        print('error! mode must be radiopropa or analytic')
        return -1
    else:
        prop = propagation.get_propagation_module(mode)
        ref_index_model = 'greenland_simple'
        ice = medium.get_ice_model(ref_index_model)
        attenuation_model = 'GL1'

        initial_point = [x_tx, 0, -z_tx]
        final_point = [x_rx, 0, -z_rx]
        rays = prop(ice, attenuation_model,
                    n_frequencies_integration=25,
                    n_reflections=0)
        rays.set_start_and_end_point(initial_point, final_point)
        rays.find_solutions()

        travel_times = []
        path_lengths = []
        amp_list = []
        n_surface = 1.3
        n_air = 1.0
        solutions = []
        for i_solution in range(rays.get_number_of_solutions()):
            # Or the path length
            path_length = rays.get_path_length(i_solution)
            # And the travel time
            travel_time = rays.get_travel_time(i_solution)
            # print('travel_time', travel_time)
            travel_times.append(travel_time)
            path_lengths.append(path_length)
            solution_int = rays.get_solution_type(i_solution)
            solution_type = solution_types[solution_int]
            solutions.append(solution_type)
            amp_of_path = 1. * (1 / path_length)  # Include Spreading Loss (missing a factor of pi)
            if i_solution == 0:
                amp_of_path *= abs(n_surface - n_air) / abs(n_surface + n_air)  # Apply Fresnel Reflection Coefficient
            amp_list.append(amp_of_path)
        return travel_times, solutions

def get_rx_arr(rxList0):
    nRx = len(rxList0)
    x_rx_arr = np.zeros(nRx)
    z_rx_arr = np.zeros(nRx)
    for i in range(nRx):
        rx_i = rxList0[i]
        x_rx_arr[i] = rx_i.x
        z_rx_arr[i] = rx_i.z
    return x_rx_arr, z_rx_arr

def get_rx_arr_unique(rxList0):
    x_rx_arr, z_rx_arr = get_rx_arr(rxList0)
    x_rx_un = np.unique(x_rx_arr)
    z_rx_un = np.unique(z_rx_arr)
    return x_rx_un, z_rx_un

def get_amp_and_t2(pulse_rx, tspace, t0 = 50, t_fluence=25, t_cut=50, factor_sinc=40, nSinc=24):
    dt = tspace[1]-tspace[0]
    fsample = 1/dt
    #print('Step 1, roll down')
    ii_max0 = np.argmax(abs(pulse_rx))
    t_max0 = tspace[ii_max0]

    ii_0 = util.findNearest(tspace, t0)
    ii_delta0 = ii_max0 - ii_0
    pulse_rx_roll = np.roll(pulse_rx, -ii_delta0)

    # Get D Maximum
    #print('D cut')
    pulse_rx_D_roll = util.cut_arr(pulse_rx_roll, tspace,
                                   t0-t_cut,
                                   t0+t_cut)
    #print('D roll')
    tspace_D_roll = util.cut_arr(tspace, tspace, t0-t_cut, t0+t_cut)
    #print('D sinc')
    tspace_sinc, pulse_rx_D_sinc_re, = util.sincInterpolateFast(tspace_D_roll,
                                               pulse_rx_D_roll,
                                               factor_sinc*fsample,
                                               nSinc)
    #print('D hilbert')
    pulse_rx_D_sinc_im = util.hilbertTransform(pulse_rx_D_sinc_re)
    pulse_rx_D_sinc = pulse_rx_D_sinc_re + 1j*pulse_rx_D_sinc_im
    iiD = np.argmax(abs(pulse_rx_D_sinc))
    tD = tspace_sinc[iiD] + (t_max0 - t0)
    ampD = abs(pulse_rx_D_sinc[iiD])
    #print('Get D maximum')
    #print('iiD', iiD, 'tD', tD, 'ampD', ampD)

    #Make R Analysis Cut
    #print('Analysis Cut')
    pulse_rx_analysis = np.roll(pulse_rx_roll, -ii_0)
    tR_low = t_fluence
    tR_high = tspace[-1] - t_fluence

    #print('R cut')
    pulse_rx_R_cut0 = util.cut_arr(pulse_rx_analysis,
                                  tspace,
                                  tR_low,
                                  tR_high)
    tspace_R_cut0 = util.cut_arr(tspace, tspace, tR_low, tR_high)
    # Find Second Maximum
    #print('Find Second Maximum')
    ii_maxR0 = np.argmax(abs(pulse_rx_R_cut0))
    tR_0 = tspace_R_cut0[ii_maxR0]

    #print('R cut')
    tspace_R_cut = util.cut_arr(tspace_R_cut0, tspace_R_cut0, tR_0-t_cut, tR_0+t_cut)
    pulse_rx_R_cut = util.cut_arr(pulse_rx_R_cut0, tspace_R_cut0, tR_0-t_cut, tR_0+t_cut)

    #print('R sinc')

    tspace_R_sinc, pulse_rx_R_sinc_re = util.sincInterpolateFast(tspace_R_cut,
                                                                 pulse_rx_R_cut,
                                                                 factor_sinc*fsample,
                                                                 nSinc)
    tspace_R_sinc += min(tspace_R_cut)
    #print('R hilbert')
    pulse_rx_R_sinc_im = util.hilbertTransform(pulse_rx_R_sinc_re)
    pulse_rx_R_sinc = pulse_rx_R_sinc_re + 1j*pulse_rx_R_sinc_im
    #print('Get R maximum')
    iiR = np.argmax(abs(pulse_rx_R_sinc))

    tR = tspace_R_sinc[iiR] + t_max0
    ampR = abs(pulse_rx_R_sinc[iiR])
    return ampD, tD, ampR, tR

path2hdf = 'data/CFM_ask_study/' #Path to paraProp HDF files
nArgs = len(argv)
if nArgs == 2:
    year_start = argv[1]
    year_end = year_start + 1
elif nArgs == 3:
    year_start = argv[1]
    year_end = argv[2]
else:
    print('Error, please enter: python ', argv[0], '<year_start> <year_end?>')
    exit()

year_l = np.arange(year_start, year_end, 1)
month_l = np.arange(1, 13, 1)

nYears = len(year_l)
nMonths = len(month_l)

datenum_l = []
datenum_str_l = []
fname_path_l, fname_hdf_l = [], []

prefix = 'nProf_CFM_'
file_suffix = '.h5'

fname_false = '999'
month_all = []
year_all = []
for i in range(nYears):
    for j in range(nMonths):
        datenum_ij = float(year_l[i]) + float(month_l[j]-0.5)/12.
        datenum_l.append(datenum_ij)
        datenum_suffix = str(year_l[i]) + '_' + str(month_l[j])
        print(datenum_ij, datenum_suffix)
        year_all.append(year_l[i])
        month_all.append(month_l[j])
        fname_hdf = prefix + datenum_suffix + file_suffix

        fname_hdf_path = join(path2hdf, fname_hdf)
        if isfile(fname_hdf_path) == True:
            fname_hdf_l.append(fname_hdf)
            fname_path_l.append(fname_hdf_path)
        else:
            fname_hdf_l.append(fname_false)
            fname_path_l.append(fname_false)

nDates = len(datenum_l)
fname_0 = fname_path_l[0]
print(fname_0)
ascan_0 = ascan()
ascan_0.load_from_hdf(fname_0)

z_tx = ascan_0.tx_depths[0] #Source Depth
tspace = ascan_0.tspace
dt = ascan_0.dt
fsample = 1/dt

rxList = ascan_0.rxList
nRx = len(rxList)
print(nRx)

#Sinc & Analysis Factors
factor = 40
nSinc = 24
t0 = 50
t_fluence = 25
t_cut = 50

x_rx_arr, z_rx_arr = get_rx_arr(rxList)
RT_arr = np.zeros((nRx,3))
rx_arr = np.zeros((nRx, 2))
rx_arr[:,0] = x_rx_arr
rx_arr[:,1] = z_rx_arr

output_labels = ['ampD', 'tD', 'ampR', 'tR']
nCount = 0

print('')
print('------------------------------------')
print('')
print('Start Scan')
duration_l = []
for ii in range(nDates):
    year_i = year_all[ii]
    month_i = month_all[ii]
    print(year_i, month_i, fname_path_l[ii])
    year_label = str(year_i)
    # Create File
    print('Create Output File')
    fname_output_hdf = 'CFM_amp_t_variance_' + str(year_i) + '.h5'
    fname_output_npy = 'CFM_amp_t_variance_' + str(year_i) + '.npy'

    if month_all[ii] == 1:
        #Create Memmap
        print('Create Memmap')
        variance_matrix = util.create_memmap(fname_output_npy, dimensions=(nDates, nRx, 4))

        print('Create HDF object')
        output_hdf = h5py.File(fname_output_hdf, 'w')
        output_hdf.create_dataset('datenum', data=datenum_l)
        output_hdf.create_dataset('rx_arr', data=rx_arr)
        output_hdf.attrs['zTx'] = z_tx
        print('Create RT array')
        output_hdf.create_dataset('RTarr', data=RT_arr)
        group = output_hdf.create_group(year_label)
        isgin = year_label in output_hdf.keys()
        print('is group', year_label, 'inside the HDF?', isgin)
        for k in range(4):
            print('Create dataset ', output_labels[k])
            ii_dec = ii + 12
            group.create_dataset(output_labels[k], data=variance_matrix[ii:ii_dec, :, k])
            print(variance_matrix[ii:ii_dec, :, k].shape, variance_matrix.shape)
        output_hdf.close()
        print('Output File', fname_output_hdf, ' Created')
    else:
        print('Output File', fname_output_hdf, ' already exists')
    print('Proceed')
    print('Check if: ', fname_path_l[ii], 'exists', fname_path_l[ii] != -999)
    if fname_path_l[ii] != '999':
        print(fname_path_l[ii], 'exists')
        ascan_i = ascan()
        ascan_i.load_from_hdf(fname_hdf=fname_path_l[ii])
        bscan_npy = ascan_i.ascan_array
        print('Loop over RxLists')

        for jj in range(nRx):
            xj = x_rx_arr[jj]
            zj = z_rx_arr[jj]
            tstart = time.time()
            pulse_rx = bscan_npy[0,jj]
            ampD, tD, ampR, tR = get_amp_and_t2(pulse_rx,
                                                tspace,
                                                t0=t0,
                                                t_fluence=t_fluence,
                                                t_cut=t_cut,
                                                factor_sinc=factor,
                                                nSinc=nSinc)
            var_l = [ampD, tD, ampR, tR]
            if jj % 10 == 0:
                print(year_all[ii], month_all[ii], xj, zj, var_l)
            for kk in range(4):
                variance_matrix[ii,jj,kk] = var_l[kk]
            tend = time.time()
            duration_s = tend-tstart
            duration_l.append(duration_s)
            nCount += 1
            if jj % 10 == 0:
                print('duration =',duration_s, 'ave duration', np.mean(duration_l), '+/-', np.std(duration_l))
                print('Remaining: ', float(nCount)/(float(nRx)*float(nDates))*100, '%')
            #Ray Tracing
            if ii == 0:
                tstart_RT = time.time()
                travel_times_j, solutions = get_ray_times(x_tx=0,z_tx=z_tx, x_rx=xj, z_rx=zj)
                print(travel_times_j, solutions)
                if len(travel_times_j) > 1:
                    RT_arr[jj, 0] = travel_times_j[1]
                    RT_arr[jj, 1] = travel_times_j[0]
                    if solutions[0] == 'reflected':
                        RT_arr[jj,2] = 1
                    elif solutions[1] == 'refracted':
                        RT_arr[jj,2] = 2
                elif len(travel_times_j) == 1:
                    RT_arr[jj,0] = travel_times_j[0]
                    RT_arr[jj,1] = -999
                    RT_arr[jj,2] = 0
                else:
                    RT_arr[jj,0], RT_arr[jj,1], RT_arr[jj,2] = -999, -999, -1
                tend_RT = time.time()
                duration_RT = tend_RT - tstart_RT
                if j % 10 == 0:
                    print('RT solutions')
                    print('duration (RT) = ', duration_RT)
            else:
                duration_RT = 0
            #Ray Tracing FInished
            t_remain_RT = duration_RT * (nRx-jj)
            t_remain = t_remain_RT + np.mean(duration_l) * (nDates - ii) * (nRx - jj)
            if jj % 10 == 0:
                print('remaining time', datetime.timedelta(seconds=t_remain))
                print('')
        print('Save to HDF file', fname_output_hdf)
        output_hdf = h5py.File(fname_output_hdf, 'r+')
        if ii == 0:
            print('Saving RT array')
            output_hdf['RTarr'][:,:] = RT_arr
        group = output_hdf[year_label]
        print('Access Group', year_label)
        for kk in range(4):
            print('Save to dataset ', output_labels[kk])
            group[output_labels[kk]][ii, :] = variance_matrix[ii, :, kk]
        output_hdf.close()
        print('Finish for', year_i, month_i)
        print('Next iteration....')
        print('')
    else:
        print('File does not exist')
        variance_matrix[ii,:,:] = -999 * np.ones((nRx, 4))