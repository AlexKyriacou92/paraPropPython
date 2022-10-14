import os
from math import factorial
import h5py
import numpy as np
import matplotlib.pyplot as pl
from random import randint, uniform
import random
from scipy.interpolate import interp1d

from genetic_operators import flat_mutation, gaussian_mutation, clone, cross_breed

path2 = 'start_profiles'
input_files = os.listdir(path2)
random.shuffle(input_files)

nProfiles = len(input_files)
print(input_files)
input_profiles = []

dz_in = 0.5
z_max = 15.0
z_min = 1.0
z_space = np.arange(z_min, z_max + dz_in, dz_in)
nOut = len(z_space)
print(nOut, z_space)

print(nProfiles, 'facotrial:', factorial(nProfiles))

for i in range(nProfiles):
    in_data = np.genfromtxt(path2 + '/' + input_files[i])
    z_prof = in_data[:,0]
    n_prof = in_data[:,1]
    
    dz = z_prof[1] - z_prof[0]
    
    f_int = interp1d(z_prof, n_prof)
    n_prof_out = np.ones(nOut)
    n_prof_out[1:-1] = f_int(z_space[1:-1])
    n_prof_out[0] = n_prof[0]
    n_prof_out[-1] = n_prof[-1]

    input_profiles.append(n_prof_out)
    print(i,len(n_prof_out))
nHalf = int(float(nProfiles)/2.)

new_profile_list = []

#Test cross breeding

for k in range(23):
    random.shuffle(input_profiles)
    for j in range(nHalf):
        i = 2*j
        n_prof_1 = input_profiles[i]
        n_prof_2 = input_profiles[i+1]
        n_prof_3 = cross_breed(n_prof_1, n_prof_2)
        new_profile_list.append(n_prof_3)
for i in range(nProfiles):
    new_profile_list.append(input_profiles[i])
'''
fig = pl.figure(figsize=(8,5),dpi=120)
ax = fig.add_subplot(111)
for k in range(nHalf):
    ax.plot(z_space, new_profile_list[k])
pl.grid()
pl.show()
'''
new_profile_arr = np.array(new_profile_list)

M = len(new_profile_arr)
print(M)
n_ave = np.ones(nOut)
n_std = np.zeros(nOut)
for i in range(nOut):
    n_i = []
    for j in range(M):
        n_i.append(new_profile_arr[j,i])
    n_ave[i] = np.mean(n_i)
    n_std[i] = np.std(n_i)

aletsch_sim = np.genfromtxt('start_profiles/aletsch_glacier_model2.txt')
fabian_data = np.genfromtxt('n_prof_PS.txt')

fig = pl.figure(figsize=(8,5),dpi=120)
ax = fig.add_subplot(111)
ax.errorbar(z_space, n_ave, n_std, xerr=0.1)
ax.plot(aletsch_sim[:,0], aletsch_sim[:,1])
ax.plot(fabian_data[:,0], fabian_data[:,1])
#ax.plot(z_space, n_ave)
#ax.fill_between(z_space, n_ave-n_std, n_ave+n_std, alpha=0.5)
ax.grid()
ax.set_xlim(0,15)
pl.show()
#Test mutation
'''
for i in range(nProfiles):
    new_profile_list.append(flat_mutation(input_profiles[i], mutation_thres=0.9))

fig = pl.figure(figsize=(8,5),dpi=120)
ax = fig.add_subplot(111)
for k in range(nProfiles):
    ax.plot(z_space, new_profile_list[k])
pl.grid()
pl.show()
'''