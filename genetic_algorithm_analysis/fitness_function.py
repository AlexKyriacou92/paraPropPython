import os
from math import factorial
import h5py
import numpy as np
import matplotlib.pyplot as pl
from random import randint, uniform
import random
from scipy.interpolate import interp1d

from genetic_operators import flat_mutation, gaussian_mutation, clone, cross_breed
import sys
#TODO: Test This

'''
def fitness_correlation(sig_sim, sig_data, mode='abs'):
    if mode == 'abs':
        if sig_sim.any() < 0:
            sig_sim = abs(sig_sim)
        elif sig_data.any() < 0:
            sig_data = abs(sig_data)
        elif sig_data.any() < 0 and sig_sim.any() < 0:
            sig_sim = abs(sig_sim)
            sig_data = abs(sig_data)
        sig_correl = sig_sim * sig_data
        S = sum(sig_correl)
    elif mode == 'real':
        if type(sig_sim.any()) == 'complex':
            sig_sim = sig_sim.real
        elif type(sig_data.any()) == 'complex':
            sig_data = sig_data.real
        elif type(sig_data.any()) == 'complex' and type(sig_sim.any()) == 'complex':
            sig_data = sig_data.real
            sig_sim = sig_sim.real
        sig_correl = abs(sig_data * sig_sim)
        S = sum(sig_correl)
    elif mode == 'complex':
        sig_correl = abs(sig_sim * sig_data)
        S = sum(sig_correl)
    else:
        print('error! enter mode = abs, real or complex')
        if sig_sim.any() < 0:
            sig_sim = abs(sig_sim)
        elif sig_data.any() < 0:
            sig_data = abs(sig_data)
        elif sig_data.any() < 0 and sig_sim.any() < 0:
            sig_sim = abs(sig_sim)
            sig_data = abs(sig_data)
        sig_correl = sig_sim * sig_data
        S = sum(sig_correl)
    return S
'''

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

def fitness_pulse_FT_data(sig_sim, sig_data):
    sig_multi = abs(sig_sim * sig_data)
    sig_multi_sq = sig_multi**2
    S = sum(sig_multi_sq)
    return S