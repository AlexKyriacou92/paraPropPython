import os
from math import factorial
import h5py
import numpy as np
import matplotlib.pyplot as pl
from random import randint, uniform
import random
from scipy.interpolate import interp1d
from math import pi

from GA_operators import flat_mutation, gaussian_mutation, clone, cross_breed
from GA_algorithm import GA
import sys
sys.path.append('../')
import util
from util import do_interpolation_same_depth, findNearest
import scipy

import sys
sys.path.append('../')
import util
import configparser
#TODO: Test This

'''
B = 1.0
C = 0.01
D = 0.5
E = 1.0
low_cut = 0.5
'''
def exp_profile(z, a, b, c):
    return a + b * np.exp(c * z)

A = 0.05
B = 1.0
C = 0.01
D = 0.5
E = 1.0
low_cut = 0.5

def makeRandomDensityVector(z, a=A, b=B, c=C, d=D, e=E, low_cut=low_cut):
    """make a vector of random density fluctuations. This is currently used with the Taylor Dome n(z) profile.
    the density fluctuations are on the order of a few percent of the density."""
    dz = abs(z[1] - z[0])
    ranVec = util.lowpassFilter(dz, low_cut, (a / (b + (z * c))) * (e*np.random.random_sample(len(z)) - d))
    return ranVec

#Initialization
def initialize_from_analytical(n_profile_mean, n_profile_std, N):
    nDepths = len(n_profile_mean)

    n_prof_list = []
    for j in range(N):
        n_prof_j = np.ones(nDepths)
        for i in range(nDepths):
            n_profile_i = np.random.normal(loc=n_profile_mean[i], scale=n_profile_std[i], size=1)
            n_prof_j[i] = n_profile_i
        n_prof_list.append(n_prof_j)
    return n_prof_list

def initalize_from_fluctuations(n_profile_mean, z_profile_mean, N, A0=0.05):
    nDepths = len(n_profile_mean)

    n_prof_list = []

    for j in range(N):
        n_prof_j = np.ones(nDepths)
        rand_vec = makeRandomDensityVector(z_profile_mean,a=A0)
        for i in range(nDepths):
            n_profile_i = n_profile_mean[i] + rand_vec[i]
            n_prof_j[i] = n_profile_i
        n_prof_list.append(n_prof_j)
    return n_prof_list

def initialize(nStart, nprofile_sampling_mean, zprofile_sampling_mean, GA, fAnalytical, fFluctuations, fFlat, fSine, fExp):
    n_prof_pool = []
    nFluctuations = int(fFluctuations * nStart)
    nAnalytical = int(fAnalytical * nStart)
    nFlat = int(fFlat * nStart)
    nSine = int(fSine * nStart)
    nExp = int(fExp * nStart)

    nprof_analytical = initialize_from_analytical(nprofile_sampling_mean, 0.08 * np.ones(GA.nGenes), 50 * nAnalytical)
    nprof_flucations = initalize_from_fluctuations(nprofile_sampling_mean, zprofile_sampling_mean, 50 * nFluctuations,A0=0.05)

    ii = 1
    jj = 1
    nAnalytical_missing = 0
    nAnalytical2 = 0
    if nAnalytical > 0:
        while ii < nAnalytical + 1:
            #print('analytical', len(nprof_analytical), nprof_analytical)
            if jj < len(nprof_analytical):
                if (np.any(nprof_analytical[jj] > 1.8) == True) or (np.any(nprof_analytical[jj] < 1.0) == True):
                    pass
                    jj += 1
                else:
                    n_prof_pool.append(nprof_analytical[jj])
                    ii += 1
                    jj += 1
                    nAnalytical2 += 1
            else:
                nAnalytical_missing = nAnalytical - nAnalytical2
                print('missing analytical: ', nAnalytical_missing)
                break

    ii = 1
    jj = 1
    '''
    print(nFluctuations)
    for i in range(nFluctuations):
        n_prof_pool.append(nprof_flucations[i])
    '''
    nFluctuations_missing = 0
    nFluctuations2 = 0
    if nFluctuations > 0:
        while ii < nFluctuations + 1:
            if jj < len(nprof_flucations):
                if (np.any(nprof_flucations[jj] > 1.8) == True) or (np.any(nprof_flucations[jj] < 1.0) == True):
                    pass
                    jj += 1
                else:
                    n_prof_pool.append(nprof_flucations[jj])
                    nFluctuations2 += 1
                    ii += 1
                    jj += 1
            else:
                nFluctuations_missing = nFluctuations - nFluctuations2
                print('missing fluctuations:', nFluctuations_missing)
                break

    ii = 1
    while ii < nFlat + 1:
        n_const = random.uniform(0, 0.78)
        n_prof_flat = np.ones(GA.nGenes) + n_const
        if (np.any(n_prof_flat > 1.8) == True) or (np.any(n_prof_flat < 1.0) == True):
            pass
        else:
            n_prof_pool.append(n_prof_flat)
            ii += 1
    ii = 1
    while ii < nSine + 1:
        n_const = random.uniform(1.0, 1.8)

        amp_rand = 0.4 * random.random()
        z_period = random.uniform(0.5, 15)
        k_factor = 1 / z_period
        phase_rand = random.uniform(0, 2 * pi)
        n_prof_sine = amp_rand * np.sin(2 * pi * zprofile_sampling_mean * k_factor + phase_rand) + n_const
        if (np.any(n_prof_sine < 1.0) == True) or (np.any(n_prof_sine > 1.8) ==True):
            pass
        else:
            n_prof_pool.append(n_prof_sine)
            ii += 1
    ii = 1
    while ii < nExp + 1:
        A = 1.78
        n0_rand = random.gauss(1.4, 0.05)
        B_rand = n0_rand - A
        C_mean = -0.04
        C_dev = 0.03
        C_rand = random.uniform(C_mean-C_dev, C_mean+C_dev)
        n_prof_exp = exp_profile(zprofile_sampling_mean, A, B_rand, C_rand)
        if (np.any(n_prof_exp < 1.0) == True) or (np.any(n_prof_exp>1.8) == True):
            pass
        else:
            n_prof_pool.append(n_prof_exp)
            ii += 1
    nMissing = nFluctuations_missing + nAnalytical_missing
    if nMissing > 0:
        ii = 1
        while ii < nExp + 1:
            A = 1.78
            n0_rand = random.gauss(1.4, 0.05)
            B_rand = n0_rand - A
            C_mean = -0.04
            C_dev = 0.03
            C_rand = random.uniform(C_mean-C_dev, C_mean+C_dev)
            n_prof_exp = exp_profile(zprofile_sampling_mean, A, B_rand, C_rand)
            if (np.any(n_prof_exp < 1.0) == True) or (np.any(n_prof_exp>1.8) == True):
                pass
            else:
                n_prof_pool.append(n_prof_exp)
                ii += 1
    random.shuffle(n_prof_pool)
    return n_prof_pool


def create_profile(zprof_out, nprof_genes, zprof_genes, nprof_override = None, zprof_override = None):
    """
    This functions creates ref-index profiles in GA
    -It combines a smoothing algorithm for the Evolving Genes
    with a fixed profile that is not acted on by the algorithm
    zprof_out : Depth Values of Output Profile
    nprof_genes :
    """
    if nprof_override is None and zprof_override is None:
        print(len(nprof_genes), len(zprof_genes))
        spi = scipy.interpolate.UnivariateSpline(zprof_genes, nprof_genes, s=0)
        nprof_out = spi(zprof_out)
    else:
        dz = zprof_out[1] - zprof_out[0]
        nDepths = len(zprof_out)
        nprof_out = np.ones(nDepths)

        zmin_2 = min(zprof_genes)
        zmax_2 = max(zprof_genes)
        ii_min2 = findNearest(zprof_out, zmin_2)

        #sp = csaps.UnivariateCubicSmoothingSpline(zprof_genes, nprof_genes, smooth=0.85)
        spi = scipy.interpolate.UnivariateSpline(zprof_genes, nprof_genes, s=0)
        zprof_2 = zprof_out[ii_min2:]
        nprof_2 = spi(zprof_2)

        nprof_out[ii_min2:] = nprof_2

        zmax_1 = max(zprof_override)
        ii_cut = findNearest(zprof_out, zmax_1)
        # zprof_1 = zprof_genes[:ii_cut]
        nDepths_1 = ii_cut

        nprof_1, zprof_1 = do_interpolation_same_depth(zprof_in=zprof_override, nprof_in=nprof_override, N=nDepths_1) #Something Set Wrong
        '''
        nprof_out[:ii_cut] = nprof_1
        spi_2 = scipy.interpolate.UnivariateSpline(zprof_out, nprof_out, s=0)
        nprof_out = spi_2(zprof_out)
        '''
        nprof_list = []
        zprof_list = []
        for i in range(len(zprof_1)):
            nprof_list.append(nprof_1[i])
            zprof_list.append(zprof_1[i])
        for j in range(len(zprof_2)):
            if zprof_2[j] > max(zprof_1):
                nprof_list.append(nprof_2[j])
                zprof_list.append(zprof_2[j])
                #print(zprof_2[j],nprof_2[j])
        spi_2 = scipy.interpolate.UnivariateSpline(zprof_list, nprof_list,s=0)
        nprof_out = spi_2(zprof_out)
    return nprof_out