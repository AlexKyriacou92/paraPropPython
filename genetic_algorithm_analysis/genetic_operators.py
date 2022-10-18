import h5py
import numpy as np
import matplotlib.pyplot as pl
from random import random, randint, uniform

#Test functions for mutation and cloning

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

def clone(prof_in):
    prof_out = prof_in
    return prof_out