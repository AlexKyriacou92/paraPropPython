import numpy as np
import math
import h5py
import sys
from sys import argv, exit
import configparser
from scipy.signal import butter, lfilter


from math import pi
from numpy import rad2deg, deg2rad, sin, cos, sqrt, log, exp, log10
import scipy.signal as sig

#import NuRadioMC
#from NuRadioMC.utilities import medium, attenuation, units

from util import *

theta_c_deg = 56.0 #Cherenkov Angle in degrees for n = 1.78, src arXiv:2208.04971
theta_c = deg2rad(theta_c_deg)

# Source: https://arxiv.org/pdf/1308.1229/1000
delta_eps_ice = 93 # Difference between DC and inf permittivity
tau_ice = 2.2e-5 # Debye relaxation time of ice
sigma_D_ice = 4.5e-3 #Conductivity of Ice at DC
eps_r_ice = 3.18 #Real Permittivity of Ice
eps0 = 8.85e-12
TeV = 1e12 # 10^12 eV = 1 TeV

def get_sigma_EM(E, f, E_lpm=2000):
    '''
    Source: Physics Letters B 411 (1997) 218-224
    :param E:
    :param f:
    :param E_lpm: - Characterisitic energy of LPM effect (~2 PeV for ice - Alvarez-Muniz, Zas 1998)
    :return:
    '''
    sigma0 = 2.7 #Opening Angle at f0, degrees
    f0 = 500.0 # standard observation frequnecy, MHz
    if E < 1e3: # 1000 TeV or 1 PeV
        sigma = sigma0*(f0/f) * E**-0.03
    else:
        sigma = sigma0*(f0/f) * (E_lpm/(0.14*E + E_lpm))**0.3
    sigma = deg2rad(sigma)
    return sigma

def get_sigma_hadronic(E, f):
    '''
    Source: astro-ph/9806098
    :param E: Shower energy (TeV)
    :param f: radio frequency [MHz]
    :return:
    '''
    sigma_d = 1.4
    f0 = 500.0 #MHz
    eps = log10(E)
    if E >= 1.0 and E < 100:
        sigma_d = (f0/f) * (2.07 - 0.33*eps + 7.5e-2 * eps**2)
    elif E >= 100.0 and E < 100e3:
        sigma_d = (f0/f) * (1.74 - 1.21e-2 * eps)
    elif E >= 100e3 and E < 10e6: # And E < 10e6 Tev or 10 EeV
        sigma_d = (f0/f) * (4.23 - 0.785*eps + 5.5e-2 * eps**2)
    sigma = deg2rad(sigma_d)
    return sigma
def E_field_on_cone(Esh, f):
    '''

    :param Esh: Shower Energy [TeV]
    :param f: Frequency [MHz]
    :return: Electric Field at 1m on the cone
    '''
    f0 = 1150 #MHz threshold frequency
    return 2.53e-7 * Esh * (f/f0) * (1 + (f/f0)**1.44)**-1

def E_field_off_cone(Esh, f, theta_v_d,mode='Hadronic'):
    '''

    :param Esh: Shower Energy [TeV]
    :param f: Frequency [MHz]
    :param theta_v_d: Viewing Angle (deg)
    :return: Electric Field at 1m off the cone
    '''
    theta_c = deg2rad(theta_c_deg)
    theta_v = deg2rad(theta_v_d)
    E0 = E_field_on_cone(Esh, f)
    if mode == 'Hadronic':
        sigma = get_sigma_hadronic(Esh, f)
    elif mode == 'EM':
        sigma = get_sigma_EM(Esh, f)
    dtheta = theta_v - theta_c
    E = E0 * (sin(theta_v)/sin(theta_c)) * exp(-1*log(2) * (dtheta/sigma)**2)
    #E = exp(-1*log(2) * (dtheta/sigma)**2)
    return E

def cond_ice(f, delta_eps, tau):
    omega = 2*pi*f
    cond = (omega**2) * tau * eps0 * delta_eps/(1 + (omega*tau)**2)
    return cond


def eps_im(f, delta_eps, tau):
    omega = 2*pi*f
    return omega*tau * delta_eps/(1+(omega*tau)**2)

def alpha(f, eps_r, eps_im):
    omega = 2*pi*f
    loss_tan = eps_im/eps_r
    c=3e8 #Speed of Light
    return omega * np.sqrt((eps_r/c**2)*(np.sqrt(1+loss_tan**2) - 1))

def create_pulse(Esh, dtheta_v,fmax=6000, df=1, t_min=-50e-3, t_max=500e-3, fs=3000,z_alpha=500.0, R_alpha=100.0, atten_model='GL1'):
    '''
    The function creates a t-domain Askaryan pulse with a 'starting' angle theta_v,
    a spectrum defined over frequency space freq, from a hadronic shower with energy Esh
    (b.d. = by default)
    The pulse is then subjected to the equivalent of R_alpha [m] (R=100 m by b.d.) of attenuation in the ice sheet,
    defined using depth z_alpha (z=500 m b.d.) and a NuMC atten-model (GL1 b.d.)

    :param Esh: Shower Energy [TeV]
    :param fmax: Maximum Observation Frequency [MHz]
    :param df: Minimum Observation Frequnecy [MHz] - used to define freq interval
    :param dtheta_v: Offset between viewing angle and Chrenkov angle [deg]
    :param t_min: Start Time used to Cut Waveform (centered at zero) [us]
    :param t_max: End Time used to Cut Waveform (centered at zero) [us]
    :param fs: Sample Frequency
    :param z_alpha: Depth used to define attenuation length [m] -> negative is below surface
    :param R_alpha: Distance used to define attenuation [m]
    :param atten_model: Refractive Index Model from nuRadioMC, used to define attenuation

    :return: time-domain pulse [V/m], tspace [us]
    '''

    #Viewing Angle
    theta_v = theta_c_deg + dtheta_v #Viewing angle, degrees

    freq = np.arange(0, fmax, df)
    nHalf = len(freq)

    # Define E-field of Askaryan at 1 m
    E_field_askyaran = E_field_off_cone(Esh, freq, theta_v)

    fspace_alpha = np.load('GL1_frequencies.npy','r')
    zspace_alpha = np.load('GL1_depths.npy', 'r')
    alpha_space = np.load('GL1_attenuation_const.npy','r') #Attenuation Constant
    ii_z = findNearest(zspace_alpha, z_alpha)
    alpha_z = alpha_space[ii_z,:]

    alpha_z_interp = np.interp(freq, fspace_alpha, alpha_z)
    Lalpha_space = np.exp(-1*alpha_z_interp*R_alpha)
    E_field_askyaran *= Lalpha_space

    #Create Symmetric Frequency Space negative frequencies
    freq_minus = -1*np.arange(df, fmax+df, df)
    freq2 = np.append(freq, np.flip(freq_minus))
    freq2_normal = np.fft.fftshift(freq2)
    nSamples = len(freq2_normal)

    E_field_askyaran2 = np.zeros(nSamples)
    E_field_askyaran2[:nHalf] = E_field_askyaran
    E_field_askyaran2[nHalf:] = np.flip(E_field_askyaran)

    #Create Output Pulse
    E_pulse_askaryan = np.fft.ifft(E_field_askyaran2)
    tspace0 = np.fft.fftfreq(nSamples, df)
    #conver to normal order [-tmax, +tmax]
    E_pulse_askaryan = np.fft.fftshift(E_pulse_askaryan)
    tspace0 = np.fft.fftshift(tspace0)
    #Reduction:
    Q_decimate = int(fmax/fs) # Make sure this is equal to an integer
    tspace0 = sig.decimate(tspace0, Q_decimate)
    E_pulse_askaryan = sig.decimate(E_pulse_askaryan, Q_decimate)
    ii_min = findNearest(tspace0, t_min)
    ii_max = findNearest(tspace0, t_max)
    tspace_cut = tspace0[ii_min:ii_max]
    E_pulse_askaryan_out = E_pulse_askaryan[ii_min:ii_max]
    t_duration = tspace_cut[-1] - tspace_cut[0]
    nSamples2 = len(tspace_cut)
    tspace_out = np.linspace(0, t_duration, nSamples2)*1e3 #Conver to ns
    return E_pulse_askaryan_out, tspace_out