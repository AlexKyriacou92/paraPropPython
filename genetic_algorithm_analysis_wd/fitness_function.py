import os
from math import factorial
import h5py
import numpy as np
import matplotlib.pyplot as pl
from random import randint, uniform
import random
from scipy.interpolate import interp1d

import sys

def inverse_signal_offset(sig_sim, sig_data, mode='abs'):
    s_ij = 0
    if mode == 'abs':
        sig_delta = abs(abs(sig_data) - abs(sig_sim))
        sig_delta = sig_delta**2
    elif mode == 'real':
        sig_delta = abs(sig_data.real - sig_sim.real)
        sig_delta = sig_delta**2
    else:
        sig_delta = abs(abs(sig_data) - abs(sig_sim))
        sig_delta = sig_delta ** 2
    inverse_s_ij = sum(sig_delta)
    s_ij = 1/inverse_s_ij
    return s_ij

def fitness_correlation(sig_sim, sig_data, mode='abs'):
    return inverse_signal_offset(sig_sim, sig_data, mode)

'''
def fitness_pulse_FT_data(sig_sim, sig_data, mode = 'Correlation'):
    if mode == 'Correlation': # Signal Cross Correlation Method
        sig_multi = abs(sig_sim * sig_data)
        sig_multi_sq = sig_multi**2
        S = sum(sig_multi_sq)
    elif mode == 'Difference': #Inverse Chi-Sq method
        sig_sim_mag = abs(sig_sim)
        sig_data_mag = abs(sig_data)
        sig_delta_sq = abs(sig_sim_mag - sig_data_mag)**2
        sum_delta_sq = sum(sig_delta_sq)
        S = 1/sum_delta_sq
    else: #Note Default is set to Correlation
        print('WARNING! \n mode should be set to: Correlation or Difference, will default to Correlation')
        sig_multi = abs(sig_sim * sig_data)
        sig_multi_sq = sig_multi ** 2
        S = sum(sig_multi_sq)
    return S
'''

def fitness_pulse_FT_data(sig_sim, sig_data, mode = 'Correlation'):
    if mode == 'Correlation':
        nSamples = len(sig_data)
        s_corr = np.zeros(nSamples)
        s_auto = np.zeros(nSamples)
        for i in range(nSamples):
            s_auto[i] = (abs(sig_sim[i]))*abs(sig_data[i])
            s_corr[i] = ((sig_sim[i].real + 1j*sig_sim[i].imag) * (sig_data[i].real - 1j*sig_data[i].imag))/s_auto[i]
        s_corr /= float(nSamples)
        s = abs(sum(s_corr))**2
    else:
        nSamples = len(sig_data)
        s_corr = np.zeros(nSamples)
        s_auto = np.zeros(nSamples)
        for i in range(nSamples):
            s_auto[i] = (abs(sig_sim[i]))*abs(sig_data[i])
            s_corr[i] = ((sig_sim[i].real + 1j*sig_sim[i].imag) * (sig_data[i].real - 1j*sig_data[i].imag))/s_auto[i]
        s_corr /= float(nSamples)
        s = abs(sum(s_corr))**2
    return s

def fitness_pulse_FT_data_2(sig_sim, sig_data, mode='Correlation'):
    nData = len(sig_data)
    S = 0
    if mode == 'Correlation': # Signal Cross Correlation Method
        dot_product = np.zeros(nData)
        for i in range(nData):
            dot_product[i] = sig_data[i].real * sig_sim[i].real
        dot_prod_sum = abs(sum(dot_product))**2 / sum(abs(sig_data))**2
        S = dot_prod_sum
    elif mode == 'Difference':
        sig_diff_c = sig_data - sig_sim
        inv_sij = 0
        for i in range(nData):
            inv_sij += abs(sig_diff_c[i])/abs(sig_data[i])
        S = 1/inv_sij**2
    else:
        print('WARNING! \n mode should be set to: Correlation or Difference, will default to Correlation')
        dot_product = np.zeros(nData)
        for i in range(nData):
            dot_product[i] = sig_data[i].real * sig_sim[i].real
        dot_prod_sum = abs(sum(dot_product)) ** 2 / sum(abs(sig_data)) ** 2
        S = dot_prod_sum
    return S