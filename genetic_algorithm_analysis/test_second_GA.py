import random

import h5py
import numpy as np
from matplotlib import pyplot as pl

from makeSim import createMatrix
from genetic_functions import initialize_from_analytical, initalize_from_fluctuations, makeRandomDensityVector
from selection_functions import roulette, selection, tournament

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
nIndividuals = 500
nGens = 60
fname_start = 'start_profiles/aletsch_glacier_2.txt'
data_nprof0 = np.genfromtxt(fname_start)
zprof_0 = data_nprof0[:,0]
nprof_0 = data_nprof0[:,1]
nDepths = len(zprof_0)

#Initialize Profiles
nHalf = int(nStart/2)
n_prof_pool = initialize_from_analytical(nprof_0, 0.04*np.ones(len(nprof_0)), int(nStart/2))

#n_prof_pool2 = np.zeros()
#n_prof_pool.append()
random.shuffle(n_prof_pool)

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
    n_prof_pop = selection(prof_list=n_prof_array[j-1], S_list=S_arr[j-1], prof_list_initial=n_prof_pool)
    #print(type(n_prof_pop), len(n_prof_pop), n_prof_pop)
    #print(j)
    n_prof_array[j] = n_prof_pop
    for i in range(nIndividuals):
        n_prof_i = n_prof_pop[i]

        S_arr[j][i] = fitness_function_prof(n_prof_test, n_prof_i)
    S_mean[j] = np.mean(S_arr[j])
    S_std[j] = np.std(S_arr[j])
    S_max[j] = np.max(S_arr[j])
    print('Gen:', j)

gen_arr = range(nGens)

fig = pl.figure(figsize=(20, 10), dpi=120)
ax1 = fig.add_subplot(221)
ax2 = fig.add_subplot(223)
ax1.errorbar(gen_arr, S_mean, S_std)
ax2.plot(gen_arr, S_max)
ax2.set_ylabel('Maximum S')
ax1.set_ylabel('Mean S')
ax2.set_xlabel('nGenerations')

ax1.grid()
ax2.grid()

j_gen = np.argmax(S_max)
print(j_gen)

ii_best = np.argmax(S_arr[j_gen])
n_prof_best = n_prof_array[j_gen][ii_best]

ii_worst = np.argmin(S_arr[j_gen])
nprof_worst = n_prof_array[j_gen][ii_worst]

ii_middle = np.argmin(abs(S_arr[j_gen]-np.median(S_arr[j_gen])))

nprof_middle = n_prof_array[j_gen][ii_middle]

ax1.scatter(gen_arr[j_gen], S_arr[j_gen][ii_best],c='b')
ax1.scatter(gen_arr[j_gen], S_arr[j_gen][ii_middle],c='g')
ax1.scatter(gen_arr[j_gen], S_arr[j_gen][ii_worst],c='r')
ax1.plot(gen_arr, S_max)


ax3 = fig.add_subplot(222)
ax4 = fig.add_subplot(224)

#ax3.plot(zprof_0, n_prof_test, c='k', label='Test Profile')
ax3.plot(zprof_0, n_prof_best, c='b', label='Best Profile: ' + str(ii_best))
#ax3.plot(zprof_0, nprof_middle, c='g', label='Middle Profile: ' + str(ii_middle))
#ax3.plot(zprof_0, nprof_worst, c='r', label='Worst Profile ' + str(ii_worst),alpha=0.5)
ax3.legend()
ax3.set_ylabel('Ref Index n')
ax3.grid()

ax4_twin = ax4.twinx()
def rho_from_n(n):
    return (n-1)/0.835

ax4.axhline(0,c='k')
ax4.plot(zprof_0, (n_prof_best - n_prof_test) / n_prof_test * 100, c='b', label='Best Profile')
ax4_twin.plot(zprof_0, (rho_from_n(n_prof_best) - rho_from_n(n_prof_test))*1e3, '--', c='b')
ax4_twin.set_ylabel(r'Density Fluctuations $\Delta \rho $ [$\mathrm{g/cm^{3}}$]')
#ax4.plot(zprof_0, (nprof_middle - n_prof_test) / n_prof_test * 100,c='g', label='Middle Profile')
#ax4.plot(zprof_0, (nprof_worst - n_prof_test) / n_prof_test *100, c='r', label='Worst Profile',alpha=0.5)
ax4.grid()
ax4.set_ylabel('Ref Index n error [%]')

ax4.set_xlabel('Z [m]')
ax4.legend()

fig.savefig('GA_S_evolution_2.png')
#pl.show()
pl.close(fig)

#Check diversity:
fig = pl.figure(figsize=(20, 10), dpi=120)
ax = fig.add_subplot()