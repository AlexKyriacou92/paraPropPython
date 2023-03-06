import os
from math import factorial
import h5py
import numpy as np
import matplotlib.pyplot as pl
from random import randint, uniform
import random
from scipy.interpolate import interp1d

from genetic_operators import flat_mutation, gaussian_mutation, clone, cross_breed, cross_breed2
def prob_from_sdist(S_ind, S_dist):
    num_lower = float(sum(S < S_ind for S in S_dist)) #count the number of individual with fitness function lower than the selected_indidivudal
    num_total = float(len(S_dist))
    P_ind = num_lower/num_total
    return P_ind 
    
    
def roulette(prof_list, S_list, nOutput):
    nIndividuals = len(S_list)
    ii = 1
    
    profile_output_list = []
    S_output_list = []
    while ii <= nOutput:
        jj_rand = random.randint(0, nIndividuals-1) # Choose Individual randomly
        
        prof_jj = prof_list[jj_rand]
        S_jj = S_list[jj_rand]
        P_jj = prob_from_sdist(S_jj, S_list)
        
        R = random.uniform(0, 1)
        if P_jj > R:
            profile_output_list.append(prof_jj)
            S_output_list.append(S_jj)
            ii += 1
    return profile_output_list, S_output_list

def tournament(prof_list, S_list, nOutput):
    nIndividuals = len(S_list)
    inds = np.array(S_list).argsort()
    random.shuffle(inds)
    
    #Shuffle the list of fitness variables and profile list
    
    S_list_tournament = S_list[inds]
    prof_list_tournament = prof_list[inds]
    
    #Subdivide into groups for tournament
    nSubgroup = int( float(nIndividuals) / float(nOutput) )
    
    profile_output_list = []
    S_output_list = []

    #Loop over each subgroup -> Select the best member and append to output list
    for i in range(nOutput):
        jj_min = i * nSubgroup
        jj_max = (i+1) * nSubgroup - 1
        S_list_subgroup = S_list_tournament[jj_min:jj_max]
        prof_list_subgroup = prof_list_tournament[jj_min:jj_max]

        #Select Best Member
        k_best = np.argmax(S_list_subgroup)
        prof_best = prof_list_subgroup[k_best]
        S_best = S_list_subgroup[k_best]

        profile_output_list.append(prof_best)
        S_output_list.append(S_best)
    return profile_output_list, S_output_list

def rand_operator(f_cross_over, f_immigrant):
    N = 1000
    N1 = int(f_cross_over * float(N))
    N2 = int(f_immigrant * float(N)) + N1


    ii_rand = random.randint(0, N-1)
    if ii_rand < N1:
        return 0 # Perform Parent Operation
    elif ii_rand >= N1 and ii_rand < N2:
        return 1 # Perform Immigrant Operation
    elif ii_rand > N2:
        return 2 #Perform Cloning Operation

def selection(prof_list, S_list, prof_list_initial, f_roulette = 0.75, f_elite = 0.01, f_cross_over = 0.72, f_immigrant = 0.22,  P_mutation = 0.01, mutation_thres = 0.95):
    '''
        prof_list : input profiles from last generation
        S_list : List of fitness scores for each member of generation
        prof_list_initial : A list containing profile sample used to initialize algorithm -> used for immigrant/injection operator

        Selection Method
        f_roulette : controls fraction of Parents selected by roulette : should be 0 < f_roulette < 1
        NOTE: f_tournmaent = 1 - f_roulette : Fraction of Parents chosen by Tournament

        Genetic Operations
        f_elite : Fraction of Profiles copied directly into next generation WITHOUT MUTATION 'Elites'
        f_cross_over : fraction of next generation created by cross_over after removing elites (can be mutated!)
        f_immigrant : fraction of next generation drawn from initial distribution through immigration after removing elites (can be mutated!)

        Note: f_clone = 1 - f_cross_over - f_Immigrant : fraction of next generation cloned into next generation (can be mutated!)

        Mutation Controls:
        P_mutation : Probablility of Profile being Mutated
        mutation_thres : Threshold (between 0 and 1) for n(z) in profile to be mutated

    '''
    nIndividuals = len(S_list) # Number of Individuals in the Generation
    nInitial = len(prof_list_initial)
    prof_list_initial_l = list(prof_list_initial)

    '''
    nParents = int(f_parent * float(nIndividuals))
    nMutants = int(f_mutants * float(nIndividuals))
    nClones = nIndividuals - nParents - nMutants #Is this redundant?? TODO: Should I have a fraction of clones pre-defined -> with the rest being filled by immigrants?
    
    #TODO: Question -> Should the likelhood of being a parent, mutant or clone be weighted??
    '''

    inds = np.array(S_list).argsort()
    inds = np.flip(inds)
    nElite = int(f_elite * float(nIndividuals))

    S_list_sorted = S_list[inds]
    prof_list_sorted = prof_list[inds]
    names = []
    S_out = []
    #TODO: Do I want to seperate elites from everyonelse
    print('Numbers:\n')
    print('nIndividuals:', nIndividuals)
    print('nElites: ', nElite)
    if nElite > 0:
        S_list_elite = S_list_sorted[:nElite]
        prof_list_elite = prof_list_sorted[:nElite]

    f_parents = 1 - f_immigrant
    nParents = int(f_parents * nIndividuals)

    ii_parent = 1
    parent_list = []

    nR = int(f_roulette * float(nParents))
    nT = nParents - nR

    roulette_list, S_list_r = roulette(prof_list, S_list, nR)
    tournament_list, S_list_t = tournament(prof_list, S_list, nT)
    for i in range(nR):
        parent_list.append(roulette_list[i])
    for j in range(nT):
        parent_list.append(tournament_list[j])

    new_generation = []
    names_common = []
    for i in range(nElite):
        new_generation.append(prof_list_elite[i])
        names.append('Elite-'+str(inds[i]))
    common_list = []
    ii_common = 1
    nCommons = nIndividuals - nElite
    while ii_common <= nCommons:
        i_operator = rand_operator(f_cross_over, f_immigrant)
        if i_operator == 0: #Cross Breeding
            p_list = random.sample(parent_list, 2)
            p1 = p_list[0]
            p2 = p_list[1]

            #prof_c = cross_breed(p1, p2)
            prof_c = cross_breed2(p1, p2)
            common_list.append(prof_c)
            names_common.append('cross-bred')
            ii_common += 1
        elif i_operator == 1:
            prof_c = random.sample(prof_list_initial_l, 1)[0]
            common_list.append(prof_c)
            names_common.append('immigrant')

            ii_common += 1
        elif i_operator == 2:
            prof_c = clone(random.sample(parent_list, 1))[0]
            common_list.append(prof_c)
            names_common.append('clone')

            ii_common += 1

    #Apply Mutations
    for i in range(nCommons):
        R = random.uniform(0,1)
        if P_mutation > R:
            #print(i, common_list[i], type(common_list[i]))
            prof_m = flat_mutation(common_list[i], mutation_thres=mutation_thres)
            names.append(names_common[i]+'-mutant')
        else:
            prof_m = common_list[i]
            names.append(names_common[i])
        new_generation.append(prof_m)
    random.shuffle(new_generation)
    #return new_generation, inds, S_list_sorted, names
    return new_generation

