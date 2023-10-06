import os
from math import factorial
import h5py
import numpy as np
import matplotlib.pyplot as pl
from random import randint, uniform
import random
from scipy.interpolate import interp1d
import names

from GA_operators import flat_mutation, gaussian_mutation, clone, cross_breed, cross_breed2


class nombre:
    def __init__(self, name):
        self.full_name = name
        self.first_name = name.split()[0]
        self.last_name = name.split()[1]


def prob_from_sdist(S_ind, S_dist):
    num_lower = float(sum(S < S_ind for S in
                          S_dist))  # count the number of individual with fitness function lower than the selected_indidivudal
    num_total = float(len(S_dist))
    P_ind = num_lower / num_total
    return P_ind


def roulette(prof_list, S_list, nOutput):
    nIndividuals = len(S_list)
    ii = 1

    profile_output_list = []
    S_output_list = []
    while ii <= nOutput:
        jj_rand = random.randint(0, nIndividuals - 1)  # Choose Individual randomly

        prof_jj = prof_list[jj_rand]
        S_jj = S_list[jj_rand]
        P_jj = prob_from_sdist(S_jj, S_list)

        R = random.uniform(0, 1)
        if P_jj > R:
            profile_output_list.append(prof_jj)
            S_output_list.append(S_jj)
            ii += 1
    return profile_output_list, S_output_list

'''
def roulette_id(prof_list, S_list, nOutput):
    nIndividuals = len(S_list)
    ii = 1

    profile_output_list = []
    S_output_list = []
    id_list = []
    jj_rand = random.shuffle()
    return profile_output_list, S_output_list, id_list
'''

def tournament(prof_list, S_list, nOutput, nSubgroup=10):
    nIndividuals = len(S_list)
    inds = np.array(S_list).argsort()
    random.shuffle(inds)

    # Shuffle the list of fitness variables and profile list

    S_list_tournament = S_list[inds]
    prof_list_tournament = prof_list[inds]

    # Subdivide into groups for tournament
    #nSubgroup = int(float(nIndividuals) / float(nOutput))

    profile_output_list = []
    S_output_list = []
    #print('nOutput', nOutput)
    #print('nSubgroup', nSubgroup)

    # Loop over each subgroup -> Select the best member and append to output list
    for i in range(nOutput):
        jj_min = 0
        jj_max = nSubgroup-1
        S_list_subgroup = S_list_tournament[jj_min:jj_max]
        #print('i output:', i)
        #print('min:', jj_min, 'max:', jj_max)
        prof_list_subgroup = prof_list_tournament[jj_min:jj_max]

        # Select Best Member
        k_best = np.argmax(S_list_subgroup)
        prof_best = prof_list_subgroup[k_best]
        S_best = S_list_subgroup[k_best]

        profile_output_list.append(prof_best)
        S_output_list.append(S_best)
        inds = np.array(S_list).argsort()
        random.shuffle(inds)

        # Shuffle the list of fitness variables and profile list
        S_list_tournament = S_list[inds]

    return profile_output_list, S_output_list

def rand_operator(f_cross_over, f_mutation):
    N = 1000
    N1 = int(f_cross_over * float(N))
    N2 = int(f_mutation * float(N)) + N1

    ii_rand = random.randint(0, N - 1)
    if ii_rand < N1:
        return 0  # Perform Parent Operation
    elif ii_rand >= N1 and ii_rand < N2:
        return 1  # Perform mutation Operation
    elif ii_rand > N2:
        return 2  # Perform Immigrant Operation

def selection(prof_list, S_list, prof_list_initial, f_roulette=0.75, f_elite=0.01, f_cross_over=0.75, f_immigrant=0.04,
              f_mutant=0.25, mutation_thres=0.95):
    nIndividuals = len(S_list)  # Number of Individuals in the Generation
    prof_list_initial_l = list(prof_list_initial)

    inds = np.array(S_list).argsort()
    inds = np.flip(inds)
    nElite = int(f_elite * float(nIndividuals))
    S_list_sorted = S_list[inds]
    prof_list_sorted = prof_list[inds]
    print('Numbers:\n')
    print('nIndividuals:', nIndividuals)
    print('nElites: ', nElite)
    if nElite > 0:
        prof_list_elite = prof_list_sorted[:nElite]
    f_parents = 1 - f_immigrant
    nParents = int(f_parents * nIndividuals)

    parent_list = []

    nR = int(f_roulette * float(nParents))
    nT = nParents - nR
    print('Roulette')
    roulette_list, S_list_r = roulette(prof_list, S_list, nR)
    print('Tournament')
    print(nT)
    tournament_list, S_list_t = tournament(prof_list, S_list, nT)
    S_parents = []
    for i in range(nR):
        parent_list.append(roulette_list[i])
        S_parents.append(S_list_r[i])
    for j in range(nT):
        parent_list.append(tournament_list[j])
        S_parents.append(S_list_t[j])
    #np.unique()
    parents_unique = []
    inds_parents_unique = np.unique(S_parents, return_index=True)[1]
    nUnique = len(inds_parents_unique)
    for i in range(nUnique):
        i_unique = inds_parents_unique[i]
        parents_unique.append(parent_list[i_unique])

    new_generation = []
    for i in range(nElite):
        new_generation.append(prof_list_elite[i])
    common_list = []
    ii_common = 1
    nCommons = nIndividuals - nElite
    print('Commons')
    while ii_common <= nCommons:
        i_operator = rand_operator(f_cross_over, f_mutant)  # TODO: Change this operator
        if i_operator == 0:  # Cross Breeding
            print('Cross-Breed')
            p_list = random.sample(parents_unique, 2) #Random Sample ->
            p1 = p_list[0]
            p2 = p_list[1]
            prof_c = cross_breed2(p1, p2)
            common_list.append(prof_c)
            ii_common += 1
        elif i_operator == 1:
            print('Mutate')

            # TODO: Add Mutation Different Methods
            j_rand = np.random.randint(0, nParents - 1, 1)[0]
            prof_m = flat_mutation(parent_list[j_rand], mutation_thres=mutation_thres)
            common_list.append(prof_m)
            ii_common += 1

        elif i_operator == 2:
            print('Immigrant')

            prof_c = random.sample(prof_list_initial_l, 1)[0]
            common_list.append(prof_c)
            ii_common += 1

    for i in range(nCommons):
        prof_m = common_list[i]
        new_generation.append(prof_m)
    return new_generation