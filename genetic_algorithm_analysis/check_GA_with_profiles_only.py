import random

import h5py
import numpy as np
from matplotlib import pyplot as pl
from scipy.interpolate import interp1d

from makeSim import createMatrix
from genetic_functions import initialize_from_analytical, roulette, tournament, initalize_from_fluctuations
import sys
sys.path.append('../')
import util
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
nIndividuals = 600
nGens = 90

fname_start = 'start_profiles/aletsch_glacier_2.txt'
#fname_test = 'start_profiles/parallel-profile-0605_2nd-pk.txt'

data_nprof0 = np.genfromtxt(fname_start)
zprof_0 = data_nprof0[:,0]
nprof_0 = data_nprof0[:,1]
nDepths0 = len(zprof_0)

#data_test = np.genfromtxt(fname_test)
#n_prof_test = data_test[:,1]

fname_glacier = 'glacier_examples/guliya.txt'
data_nprof_glacier = np.genfromtxt(fname_glacier)
zprof_glacier0 = data_nprof_glacier[:, 0]
nprof_glacier0 = data_nprof_glacier[:, 1]
f_prof_interp = interp1d(zprof_0, nprof_0)
func_interp_glacier = interp1d(zprof_glacier0, nprof_glacier0)

dz_start = zprof_0[1] - zprof_0[0]
dz_small = 0.2


if dz_small != dz_start:
    factor = int(dz_start/dz_small)

    zprof_1 = np.arange(min(zprof_0), max(zprof_0) + dz_small, dz_small)
    nDepths = len(zprof_1)

    nprof_1 = np.ones(nDepths)
    nprof_1[0] = nprof_0[0]
    nprof_1[-1] = nprof_0[-1]
    nprof_1[1:-1] = f_prof_interp(zprof_1[1:-1])
    nprof_glaicer = np.ones(nDepths)
    nprof_glacier = func_interp_glacier(zprof_1)

    n_prof_test = nprof_glacier

else:
    zprof_1 = zprof_0
    nprof_1 = nprof_0
    nDepths = len(zprof_1)
    nprof_glaicer = np.ones(nDepths)
    nprof_glacier = func_interp_glacier(zprof_1)
    n_prof_test = nprof_glacier

#Initialize Profiles

nHalf = nStart//4

n_prof_pool = initialize_from_analytical(nprof_1, 0.04*np.ones(len(nprof_1)), nHalf)
n_prof_pool2 = initalize_from_fluctuations(nprof_1, zprof_1, nHalf)

for i in range(nHalf):
    n_prof_pool.append(n_prof_pool2[i])

for i in range(nHalf):
    n_const = 0.8 * random.random()
    n_prof_flat = np.ones(nDepths) + n_const
    n_prof_pool.append(n_prof_flat)
from math import pi
for i in range(nHalf):
    amp_rand = 0.4*random.random()
    z_period = random.uniform(0.5,15)
    k_factor = 1/z_period
    phase_rand = random.uniform(0, 2*pi)
    freq_rand = amp_rand*np.sin(2*pi*zprof_0*k_factor + phase_rand)
    n_prof_flat = np.ones(nDepths) + n_const
    n_prof_pool.append(n_prof_flat)
random.shuffle(n_prof_pool)
n_prof_initial = n_prof_pool[:nIndividuals]

S_arr = np.zeros((nGens, nIndividuals))
n_prof_array = np.ones((nGens, nIndividuals, nDepths))

print(len(n_prof_test), len(n_prof_initial[0]))
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
fig.suptitle('GA Results, $N_{pop}$ =' + str(nIndividuals) + ', $N_{gens}$ =' + str(nGens))
ax1 = fig.add_subplot(221)
ax2 = fig.add_subplot(223)
ax1.set_title('Fitness Score evolution')
ax2.set_title('Max Fitness Score')
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
ax3.set_title('Ref Index Reconstruction')
ax4.set_title('Ref Index Resiuals (best result)')
ax3.plot(zprof_1, nprof_1, c='r')
ax3.plot(zprof_1, n_prof_test, c= 'k',label='Test Distribution')
ax3.plot(zprof_1, n_prof_best, c='b', label='Best Distribution')
ax3.plot(zprof_glacier0, nprof_glacier0,c='g')
#ax3.plot(zprof_1, nprof_rand_first,c='g', label='From inital sample')
#ax3.plot(zprof_1, nprof_rand_last, c='r', label='From last generation (random)')
ax3.legend()
ax3.set_ylabel('Ref Index n')
ax3.grid()

def n2rho(n): # kg / m^3
    return (n-1)/0.835 * 1e3

n_residuals = n_prof_best - n_prof_test
n_residuals_rel = n_residuals/n_prof_test

print('ref index residuals')
print(np.mean(n_residuals), np.std(n_residuals))


rho_best = n2rho(n_prof_best)
rho_test = n2rho(n_prof_test)
rho_residuals = rho_best - rho_test
print('density residuals')
print(np.mean(rho_residuals), np.std(rho_residuals))

ax5 = ax4.twinx()
ax4.plot(zprof_1, n_residuals, c='b', label='Best Distribution (residuals)')
ax5.plot(zprof_1, rho_residuals, c='r',label='Density residuals')
ax5.set_ylabel(r'$\Delta \rho $ [$\mathrm{kg/m^{3}}$]')
#ax4.plot(zprof_1, (nprof_rand_first - n_prof_test) / n_prof_test * 100,c='g', label='From inital sample')
#ax4.plot(zprof_1, (nprof_rand_last - n_prof_test) / n_prof_test *100, c='r', label='From last generation (random)')
ax4.grid()
ax4.set_ylabel(r'Ref index residuals $\Delta n$')

ax4.set_xlabel('Z [m]')
ax4.legend()

fig.savefig('GA_S_evolution.png')
pl.show()
#pl.close(fig)