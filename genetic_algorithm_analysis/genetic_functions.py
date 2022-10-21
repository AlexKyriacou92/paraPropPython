import os
from math import factorial
import h5py
import numpy as np
import matplotlib.pyplot as pl
from random import randint, uniform
import random
from scipy.interpolate import interp1d

from genetic_operators import flat_mutation, gaussian_mutation, clone, cross_breed

#TODO: Test This

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

def initalize_from_fluctuations(n_profile_mean, rand_vec, N):
    nDepths = len(n_profile_mean)

    n_prof_list = []
    for j in range(N):
        n_prof_j = np.ones(nDepths)
        for i in range(nDepths):
            n_profile_i = n_profile_mean[i] + rand_vec[i]
            n_prof_j[i] = n_profile_i
        n_prof_list.append(n_prof_j)
    return n_prof_list

#TODO: Save these to h5 file matrix

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

def roulette(pop_list, S_list, nprof_initial, clone_fraction = 0.1, parent_fraction = 0.8, immigrant_fraction = 0.1, mutation_thres = 0.95, mutation_prob = 0.5):
    '''
    pop_list : list of n_profiles from generation
    S_list : the fitness function from generation
    nprof_initial : the initial distribution of n_profiles

    '''
    X = np.array(pop_list)
    Y = np.array(S_list)
    #Z = [x for _, x in sorted(Y,X)]
    inds = Y.argsort()
    Z = X[inds]
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

    nImmigrants = N - nChildren - 1
    jj_ints = np.random.randint(0, len(nprof_initial), nImmigrants)
    for m in range(nImmigrants):
        r = random.uniform(0, 1)
        if r > mutation_prob:
            print(nprof_initial[jj_ints[m]])
            new_population.append(flat_mutation(nprof_initial[jj_ints[m]],mutation_thres=mutation_thres))
        else:
            new_population.append(nprof_initial[jj_ints[m]])

    print('new population length:',len(new_population))
    return new_population