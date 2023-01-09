import os
from math import factorial
import h5py
import numpy as np
import matplotlib.pyplot as pl
from random import randint, uniform
import random
from scipy.interpolate import interp1d
from math import pi

from genetic_operators import flat_mutation, gaussian_mutation, clone, cross_breed
from genetic_algorithm import GA
import sys
sys.path.append('../')
import util

import sys
sys.path.append('../')
import util
import configparser
#TODO: Test This
B = 1.0
C = 0.01
D = 0.5
E = 1.0
low_cut = 0.5

def exp_profile(z, a, b, c):
    return a + b * np.exp(c * z)

def makeRandomDensityVector(z, a=0.6, b=B, c=C, d=D, e=E, low_cut=low_cut):
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

def initalize_from_fluctuations(n_profile_mean, z_profile_mean, N):
    nDepths = len(n_profile_mean)

    n_prof_list = []

    for j in range(N):
        n_prof_j = np.ones(nDepths)
        rand_vec = makeRandomDensityVector(z_profile_mean)
        for i in range(nDepths):
            n_profile_i = n_profile_mean[i] + rand_vec[i]
            n_prof_j[i] = n_profile_i
        n_prof_list.append(n_prof_j)
    return n_prof_list
#TODO: Save these to h5 file matrix


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

def initialize(nStart, nprofile_sampling_mean, zprofile_sampling_mean, GA_1, fAnalytical, fFluctuations, fFlat, fSine, fExp):
    n_prof_pool = []
    nFluctuations = int(fFluctuations * nStart)
    nAnalytical = int(fAnalytical * nStart)
    nFlat = int(fFlat * nStart)
    nSine = int(fSine * nStart)
    nExp = int(fExp * nStart)

    nprof_analytical = initialize_from_analytical(nprofile_sampling_mean, 0.08 * np.ones(GA_1.nGenes), 2 * nAnalytical)
    nprof_flucations = initalize_from_fluctuations(nprofile_sampling_mean, zprofile_sampling_mean, 2 * nFluctuations)

    ii = 1
    while ii < nAnalytical + 1:
        if any(nprof_analytical[ii]) > 1.8 or any(nprof_analytical[ii]) < 1.0:
            pass
        else:
            n_prof_pool.append(nprof_analytical[ii])
            ii += 1
    ii = 1
    while ii < nFluctuations + 1:
        if any(nprof_flucations[ii]) > 1.8 or any(nprof_flucations[ii]) < 1.0:
            pass
        else:
            n_prof_pool.append(nprof_flucations[ii])
            ii += 1
    ii = 1
    while ii < nFlat + 1:
        n_const = random.uniform(0, 0.78)
        n_prof_flat = np.ones(GA_1.nGenes) + n_const
        if any(n_prof_flat) > 1.8 or any(n_prof_flat) < 1.0:
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
        if any(n_prof_sine) < 1.0 or any(n_prof_sine) > 1.8:
            pass
        else:
            n_prof_pool.append(n_prof_sine)
            ii += 1
    ii = 1
    while ii < nExp + 1:
        B_rand = random.uniform(-1, -0.01)
        C_rand = random.uniform(-0.03, -0.005)
        n_prof_exp = exp_profile(zprofile_sampling_mean, 1.78, B_rand, C_rand)
        if any(n_prof_exp) < 1.0 or any(n_prof_exp) > 1.8:
            pass
        else:
            n_prof_pool.append(n_prof_exp)
            ii += 1
    random.shuffle(n_prof_pool)
    return n_prof_pool
#Genetic Opeators -> work as if the population is 100

#=======================================================
#Sort and select
#=======================================================

def tournament(pop_list, S_list, nprof_initial, clone_fraction = 0.1, parent_fraction = 0.8, immigrant_fraction = 0.1, mutation_thres = 0.95, mutation_prob = 0.5):
    random.shuffle(pop_list)
    N = len(pop_list)
    R = 10
    M = int(float(N)/R)

    Norm = clone_fraction + parent_fraction + immigrant_fraction
    if Norm != 1.0:
        clone_fraction /= Norm
        parent_fraction /= Norm
        immigrant_fraction /= Norm

    new_population = []
    for n in range(M):
        jj1 = n*R
        jj2 = (n+1)*R
        pop_list_sub = pop_list[jj1:jj2]
        S_list_sub = S_list[jj1:jj2]

        '''
        X = pop_list_sub
        Y = S_list_sub
        Z = [x for _, x in sorted(Y, X)]
        '''
        X = np.array(pop_list_sub)
        Y = np.array(S_list_sub)

        inds = Y.argsort()
        Z = X[inds]
        N = len(Z)

        ii_clone = int(clone_fraction * float(R))
        ii_parent = int(parent_fraction * float(R))

        clone_list = Z[:ii_clone]
        parent_list = Z[ii_clone:ii_parent]

        nClones = len(clone_list)
        for i in range(nClones):
            nprof_i = clone_list[i]
            nprof_new = clone(nprof_i)
            new_population.append(nprof_new)

        nParents = len(parent_list)
        nCouples = int(float(nParents) / 2.)
        children_list0 = []

        # breed children, two parents, one child
        for j in range(nCouples):
            k = 2 * j
            n_prof_p = parent_list[k]
            n_prof_m = parent_list[k + 1]
            n_prof_c = cross_breed(n_prof_p, n_prof_m)
            children_list0.append(n_prof_c)

        # shuffle parents, breed more children
        random.shuffle(parent_list)
        for j in range(nCouples):
            k = 2 * j
            n_prof_p = parent_list[k]
            n_prof_m = parent_list[k + 1]
            n_prof_c = cross_breed(n_prof_p, n_prof_m)
            children_list0.append(n_prof_c)

        # apply mutations
        children_list = []

        for k in range(nParents):
            nprof_k = children_list0[k]
            r = random.uniform(0, 1)
            if r > mutation_prob:
                prof_mutant = flat_mutation(nprof_k, mutation_thres=mutation_thres)
            else:
                prof_mutant = clone(nprof_k)
            children_list.append(prof_mutant)
        nChildren = len(children_list)

        for l in range(nChildren):
            new_population.append(children_list[l])

        nImmigrants = R - nChildren
        jj_ints = np.random.randint(0, len(nprof_initial), nImmigrants)
        for m in range(nImmigrants):
            r = random.uniform(0, 1)
            if r > mutation_prob:
                new_population.append(flat_mutation(nprof_initial[jj_ints[m]], mutation_thres=mutation_thres))
            else:
                new_population.append(nprof_initial[jj_ints[m]])
    return new_population

def roulette(nprof_list, S_list, nprof_pool, clone_fraction = 0.1, children_fraction = 0.8, immigrant_fraction = 0.1, mutation_thres = 0.95, mutation_prob = 0.5):
    new_population = []

    inds = np.array(S_list).argsort()
    inds = np.flip(inds)

    nprof_list = np.array(nprof_list)

    S_list_sorted = S_list[inds]
    nprof_list = nprof_list[inds]

    nIndividuals = len(inds)

    nHalf = int(float(nIndividuals)/2.)
    nOtherHalf = nIndividuals - nHalf

    nprof_list_cut = nprof_list[:nHalf]
    Norm = clone_fraction + children_fraction + immigrant_fraction
    if Norm != 1.0:
        clone_fraction /= Norm
        children_fraction /= Norm
        immigrant_fraction /= Norm

    nClones = int(clone_fraction * nIndividuals) #Number of 'cloned profiles' -> the proportion is the same before and afte
    nChildren = int(children_fraction * nIndividuals)
    nParents = int(float(nChildren)/2)
    nCouples = int(float(nParents)/2)

    print('clones:', nClones)
    print('couples:', nCouples)
    print('parents:', nParents)
    print('children:', nChildren)

    nImmigrants = nIndividuals - nClones - nChildren
    print(nImmigrants)
    print('sum of clones, children and migrants:', nImmigrants + nClones + nChildren, ' should equal: ', nIndividuals)

    #Step 1 -> Select the top fraction (by default the top 10% of S scores) and clone them!
    #Divide this into two subgroups -> the Elite will not be mutated, the
    clone_list0 = nprof_list_cut[:nClones]
    parent_list = nprof_list_cut[nClones:]
    print(len(parent_list))

    nElite = int(float(nClones)/2.)
    clone_list = []
    for i in range(nElite):
        clone_list.append(clone(clone_list0[i]))
    for j in range(nElite, nClones):
        nprof_clone = clone(clone_list0[j])
        r = random.uniform(0,1)
        if r > mutation_prob:
            nprof_mutant = flat_mutation(nprof_clone,mutation_thres=mutation_thres)
        else:
            nprof_mutant = nprof_clone
        clone_list.append(nprof_mutant)

    for l in range(nClones):
        new_population.append(clone_list[l])

    #Step 2 -> the Middle Group (by defult the group with: top 10 % > S > bottom 10% or the middle 80% from the surviving half) -> breed them
    children_list = []
    print(len(parent_list), 2*nCouples)
    for k in range(4):
        for i in range(nCouples):
            j = 2*i
            nprof_p = parent_list[j]
            nprof_m = parent_list[j+1]
            n_prof_c = cross_breed(nprof_p, nprof_m)
            children_list.append(n_prof_c)
        random.shuffle(parent_list)

    for l in range(nChildren):
        r = random.uniform(0,1)
        nprof_c = children_list[l]
        if r > mutation_prob:
            nprof_m = flat_mutation(nprof_c, mutation_thres=mutation_thres)
        else:
            nprof_m = nprof_c
        new_population.append(nprof_m)

    nprof_pool_in = nprof_pool
    random.shuffle(nprof_pool_in)

    for i in range(nImmigrants):
        nprof_i = nprof_pool_in[i]
        r = random.uniform(0,1)
        if r > mutation_prob:
            nprof_m = flat_mutation(nprof_i, mutation_thres=mutation_thres)
        else:
            nprof_m = nprof_i
        new_population.append(nprof_m)

    return new_population



'''     
def roulette(pop_list, S_list, nprof_initial, clone_fraction = 0.1, parent_fraction = 0.8, immigrant_fraction = 0.1, mutation_thres = 0.95, mutation_prob = 0.5):

    #pop_list : list of n_profiles from generation
    #S_list : the fitness function from generation
    #nprof_initial : the initial distribution of n_profiles
    #print('input:', S_list)

    X = np.array(pop_list)
    Y = np.array(S_list)
    #Z = [x for _, x in sorted(Y,X)]
    inds = Y.argsort()
    inds = np.flip(inds)
    Z = X[inds]    
    
    #print('sorted:',Y[inds], '\n', X[inds])
    N = len(Z)

    Norm = clone_fraction + parent_fraction + immigrant_fraction
    if Norm != 1.0:
        clone_fraction /= Norm
        parent_fraction /= Norm
        immigrant_fraction /= Norm
    ii_clone = int(clone_fraction * float(N))
    ii_parent = int(parent_fraction * float(N))


    clone_list = Z[:ii_clone]
    parent_list = Z[ii_clone:ii_parent]

    new_population = []

    nClones = len(clone_list)
    for i in range(nClones):
        nprof_i = clone_list[i]
        nprof_new = clone(nprof_i)
        new_population.append(nprof_new)

    nParents = len(parent_list)
    nCouples = int(float(nParents)/2.)
    children_list0 = []

    #breed children, two parents, one child
    for j in range(nCouples):
        k = 2*j
        n_prof_p = parent_list[k]
        n_prof_m = parent_list[k+1]
        n_prof_c = cross_breed(n_prof_p, n_prof_m)
        children_list0.append(n_prof_c)

    #shuffle parents, breed more children
    random.shuffle(parent_list)
    for j in range(nCouples):
        k = 2*j
        n_prof_p = parent_list[k]
        n_prof_m = parent_list[k+1]
        n_prof_c = cross_breed(n_prof_p, n_prof_m)
        children_list0.append(n_prof_c)

    #apply mutations
    children_list = []

    for k in range(len(children_list0)):
        nprof_k = children_list0[k]
        r = random.uniform(0, 1)
        if r > mutation_prob:
            prof_mutant = flat_mutation(nprof_k, mutation_thres=mutation_thres)
        else:
            prof_mutant = clone(nprof_k)
        children_list.append(prof_mutant)
    nChildren = len(children_list)

    for l in range(nChildren):
        new_population.append(children_list[l])
    #print(N, 'old population')
    #print('children and clones',nChildren + nClones)
    nImmigrants = N - nChildren - nClones
    #print('immigrants:', nImmigrants)
    jj_ints = np.random.randint(0, len(nprof_initial), nImmigrants)
    for m in range(nImmigrants):
        r = random.uniform(0, 1)
        if r > mutation_prob:
            #print(nprof_initial[jj_ints[m]])
            new_population.append(flat_mutation(nprof_initial[jj_ints[m]],mutation_thres=mutation_thres))
        else:
            new_population.append(nprof_initial[jj_ints[m]])

    #print('new population length:',len(new_population))
    #print('old population:', N)
    return new_population
'''