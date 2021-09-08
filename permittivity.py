from numpy import sqrt
from math import pi
import numpy as np

#Fundamental Constants
epsilon0 = 8.85e-12 #Permittivity of Free Space
mu0 = 4*pi*1e-7 #Permeability of Free Space
c0 = 1./sqrt(epsilon0*mu0) #Speed of Light in Vacuum

#Functions
def eps2m(eps_c): #Return complex refractive index (m) from complex relative permittivity (eps_c)
    eps_r = eps_c.real
    eps_i = eps_c.imag
    n_sq = (1/2.) * (sqrt(eps_r**2 + eps_i**2) + eps_r )
    k_sq = (1/2.) * (sqrt(eps_r**2 + eps_i**2) - eps_r )
    n = sqrt(n_sq)
    k = sqrt(k_sq)
    m = n + 1j*k
    return m

def m2eps(m):
    eps_c = m**2
    return eps_c

def cond2eps_im(sigma, f):
    w = 2*pi*f
    return sigma/(w*epsilon0)

def sigma_from_p(p, sigma): #p: porosity, sigma: conductivity for p = 0
    v = 1 - p #volume filling factor
    return sigma*v*(0.68 + 0.32*v)**2

def alpha(eps_c, f):
    m_c = eps2m(eps_c)
    k = m_c.imag
    w = 2*pi*f
    return 4*w*k/c0


def pure_ice(x, z):
    n_vacuum = 1.
    n_ice = 1.78 + 1e-5j
    n_material = n_vacuum
    if z >= 0:
        n_material = n_ice
    else:
        n_material = n_vacuum
    return n_material

def southpole(z):
    A=1.78
    B=-0.43
    C=-0.0132
    return A+B*np.exp(C*z)

def rho2n(rho):
    eps_r = (1 + 0.835*rho)**2
    return eps2m(eps_r)

def poro2n(poro,eps_r0):
    eps_r = eps_r0*poro*(0.68 + 0.32*poro)**2
    return eps2m(eps_r)

sigma_solid_pure_ice = 8.7e-6 #uS/m
P_surface = 0.9
sigma_pure_snow = sigma_from_p(P_surface, sigma_solid_pure_ice)
#print(sigma_pure_snow)

freq_test = 500e6 #100 MHz
eps_i_ice = cond2eps_im(sigma_solid_pure_ice, freq_test)
print(eps_i_ice)

eps_snow = 1.2 + 0.015j #Permittivity of Enceladus Snow (Type III)
m_snow = eps2m(eps_snow) #

eps_ice = 3.2 + 1j*eps_i_ice
m_ice = eps2m(eps_ice)
#print('Snow, eps_r =', eps_snow, 'n = ', m_snow, 'alpha = ', alpha(eps_snow, freq_test))
#print('Ice, eps_r =', eps_ice, 'n =', m_ice, 'alpha = ', alpha(eps_ice, freq_test))

eps_meteor = 8.2 + 0.1558j #Taken from Herique et al. (2018) Direct Observations of Asteroid Interior and Regolith Structure: Science Measurement Requirements
eps_vacuum = 1.0
eps_water = 82 + 899j #Based on the Conductivity of Salt Water -> 5 S/m
def enceladus_2layer(z, snow_depth=100): #Create a flat layer of snow above ice
    n_snow = eps2m(eps_snow)
    n_ice = eps2m(eps_ice)
    n_vacuum = 1.0
    n_material = n_vacuum

    if z >= 0 and z < snow_depth:
        n_material = n_snow
    elif z >= snow_depth:
        n_material = n_ice
    return n_material

def enceladus_environ(x, z, snow_depth = 100, meteor_list = [], crevass_list = [], aquifer_list=[]): #Creates a 2 layer geometry with added meteorites (spheres), crevasses and aquifers (triangles)
    n_medium = enceladus_2layer(z, snow_depth)

    numMeteors = len(meteor_list)
    numCrevasses = len(crevass_list)
    numAquifers = len(aquifer_list)
    #Loop over meteorites


    for i in range(numMeteors):
        meteor_i = meteor_list[i] #must be sphere
        if meteor_i.isInside(x,z) == True:
            n_medium = eps2m(meteor_i.eps_r)

    for i in range(numCrevasses):
        crevass_i = crevass_list[i]
        if crevass_i.isInside(x,z) == True:
            n_medium = eps2m(crevass_i.eps_r)

    for i in range(numAquifers):
        aquifer_i = aquifer_list[i]
        if aquifer_i.isInside(x,z) == True:
            n_medium = eps2m(aquifer_i.eps_r)

    return n_medium