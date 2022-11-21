import random

import h5py
import numpy as np
from matplotlib import pyplot as pl

from makeSim import createMatrix
from genetic_functions import initialize_from_analytical, roulette, tournament

def fitness_function_prof(n_test, n_base):
    nDepths = len(n_base)

    sum_dn_sq = 0
    for i in range(nDepths):
        dn = abs(n_test[i] - n_base[i])
        dn_sq = dn**2
        sum_dn_sq += dn_sq
    S = 1/sum_dn_sq
    return S

nStart = 10000
nIndividuals = 200
nGens = 60
fname_start = 'start_profiles/aletsch_glacier_2.txt'
data_nprof0 = np.genfromtxt(fname_start)
zprof_0 = data_nprof0[:,0]
nprof_0 = data_nprof0[:,1]
nDepths = len(zprof_0)

#Initialize Profiles


n_prof_pool = initialize_from_analytical(nprof_0, 0.04*np.ones(len(nprof_0)), nStart)
n_prof_initial = n_prof_pool[:nIndividuals]

n_prof_test = nprof_0

S_arr = np.zeros((nGens, nIndividuals))
n_prof_array = np.ones((nGens, nIndividuals, nDepths))
for i in range(nIndividuals):
    n_prof_i = n_prof_initial[i]

    S_arr[0][i] = fitness_function_prof(n_prof_test, n_prof_i)
    n_prof_array[0][i] = n_prof_i

S_mean = np.zeros(nGens)
S_std = np.zeros(nGens)
S_max = np.zeros(nGens)
S_mean[0] = np.mean(S_arr[0])
S_std[0] = np.std(S_arr[0])
S_max[0] = np.max(S_arr[0])

for j in range(1, nGens):
    n_prof_pop = roulette(n_prof_array[j-1], S_arr[j-1], n_prof_pool, clone_fraction=0.1, children_fraction=0.8, immigrant_fraction=0.1)
    #n_prof_pop = tournament(n_prof_array[j-1], S_arr[j-1], n_prof_pool, clone_fraction=0.1, parent_fraction=0.85, immigrant_fraction=0.05)
    n_prof_array[j] = n_prof_pop
    for i in range(nIndividuals):
        n_prof_i = n_prof_pop[i]

        S_arr[j][i] = fitness_function_prof(n_prof_test, n_prof_i)
    S_mean[j] = np.mean(S_arr[j])
    S_std[j] = np.std(S_arr[j])
    S_max[j] = np.max(S_arr[j])
    print('Gen:', j)

gen_arr = range(nGens)

fig = pl.figure(figsize=(20,10),dpi =120)
ax1 = fig.add_subplot(221)
ax2 = fig.add_subplot(223)
ax1.errorbar(gen_arr, S_mean, S_std)
ax2.plot(gen_arr, S_max)
ax2.set_ylabel('Maximum S')
ax1.set_ylabel('Mean S')
ax2.set_xlabel('nGenerations')

ax1.grid()
ax2.grid()


ii_best = np.argmax(S_arr[-1])
n_prof_best = n_prof_array[-1][ii_best]
ii_rand = random.randint(0,nIndividuals-1)
nprof_rand_last = n_prof_array[-1][ii_rand]

ii_rand2 = random.randint(0,nIndividuals-1)
nprof_rand_first = n_prof_pool[ii_rand2]

ax3 = fig.add_subplot(222)
ax4 = fig.add_subplot(224)

ax3.plot(zprof_0, n_prof_test, c= 'k',label='Test Distribution')
ax3.plot(zprof_0, n_prof_best, c='b', label='Best Distribution')
ax3.plot(zprof_0, nprof_rand_first,c='g', label='From inital sample')
ax3.plot(zprof_0, nprof_rand_last, c='r', label='From last generation (random)')
ax3.legend()
ax3.set_ylabel('Ref Index n')
ax3.grid()

def n2rho(n): # kg / m^3
    return (n-1)/0.835 * 1e3

n_residuals = n_prof_best - n_prof_test
n_residuals_rel = n_residuals/n_prof_test

rho_best = n2rho(n_prof_best)
rho_test = n2rho(n_prof_test)
rho_residuals = rho_best - rho_test

ax5 = ax4.twinx()
ax4.plot(zprof_0, n_residuals, c='b', label='Best Distribution (residuals)')
ax5.plot(zprof_0, rho_residuals, c='r',label='Density residuals')
ax5.set_ylabel(r'$\Delta \rho $ [$\mathrm{kg/m^{3}}$]')
#ax4.plot(zprof_0, (nprof_rand_first - n_prof_test) / n_prof_test * 100,c='g', label='From inital sample')
#ax4.plot(zprof_0, (nprof_rand_last - n_prof_test) / n_prof_test *100, c='r', label='From last generation (random)')
ax4.grid()
ax4.set_ylabel('Ref Index n error [%]')

ax4.set_xlabel('Z [m]')
ax4.legend()

fig.savefig('GA_S_evolution.png')
pl.show()
#pl.close(fig)