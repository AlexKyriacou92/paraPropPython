

import h5py
import numpy as np
import matplotlib.pyplot as pl
from random import random, randint, uniform
#from genetic_functions import makeRandomDensityVector
#Test functions for mutation and cloning
import sys
sys.path.append('../')
import util
B = 1.0
C = 0.01
D = 0.5
E = 1.0
low_cut = 0.5
def makeRandomDensityVector(z, a=0.6, b=B, c=C, d=D, e=E, low_cut=low_cut):
    """make a vector of random density fluctuations. This is currently used with the Taylor Dome n(z) profile.
    the density fluctuations are on the order of a few percent of the density."""
    dz = abs(z[1] - z[0])
    ranVec = util.lowpassFilter(dz, low_cut, (a / (b + (z * c))) * (e*np.random.random_sample(len(z)) - d))
    return ranVec

def cross_breed(prof1, prof2):
    '''
        This function takes input profiles 1 and 2 'parents' and produces profile 3 'child'
        Each point in profile 3 is randomly sampled from profile 1 or 2 with equal probability
    '''
    N = len(prof1)
    prof3 = np.ones(N)

    for i in range(N):
        r = random()
        if r < 0.5:
            prof3[i] = prof1[i]
        elif r >= 0.5:
            prof3[i] = prof2[i]
    return prof3

def cross_breed2(prof1, prof2):
    N = len(prof1)
    j_cut = randint(0, N-1)
    prof3 = np.ones(N)
    prof3[:j_cut] = prof1[:j_cut]
    prof3[j_cut:] = prof2[j_cut:]
    return prof3

def gaussian_mutation(prof_in, mutation_thres, n_prof_mean, n_prof_var):
    N = len(prof_in)
    M = 100
    prof_out = np.ones(N)
    for i in range(N):
        r = random()
        if r > mutation_thres:
            jj = randint(0, M-1)
            gaussian_dist = np.random.normal(n_prof_mean[i], n_prof_var[i], M)
            prof_out[i] = gaussian_dist[jj]
        else:
            prof_out[i] = prof_in[i]
    return prof_out

def flat_mutation(prof_in, mutation_thres, nmin = 1.2, nmax=1.8):
    N = len(prof_in)
    prof_out = np.ones(N)

    for i in range(N):
        r = random()
        if r > mutation_thres:
            prof_out[i] = uniform(nmin, nmax)
        else:
            prof_out[i] = prof_in[i]
    return prof_out

def fluctuation_mutation(prof_in, mutation_thres, mutation_var):
    #TODO: Change Mutation Method
    N = len(prof_in)
    prof_out = np.ones(N)
    M = 100
    for i in range(N):
        r = random()
        if r > mutation_thres:
            gaussian_dist = np.random.normal(prof_in[i], mutation_var, M)
            ii = 1
            n_ii = 1
            while ii < 2:
                jj = randint(0, M-1)
                n_jj = gaussian_dist[jj]
                if n_jj < 1.8 and n_jj > 1.2:
                    n_ii = n_jj
                    ii += 1
            prof_out[i] = n_ii
        else:
            prof_out[i] = prof_in[i]
    return prof_out


def mutation(prof_in, mutation_thres, z_space, i_mode=0):
    N = len(prof_in)
    prof_out = np.ones(N)
    if i_mode == 0:
        ranVec = makeRandomDensityVector(z_space)
        for i in range(N):
            r = random()
            if r > mutation_thres:
                prof_out[i] = prof_in[i] + ranVec[i]
            else:
                prof_out[i] = prof_in[i]
        return prof_out
    elif i_mode == 1:
        prof_out = flat_mutation(prof_in, mutation_thres)
        return prof_out

def clone(prof_in):
    prof_out = prof_in
    return prof_out