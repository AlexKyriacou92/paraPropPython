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
