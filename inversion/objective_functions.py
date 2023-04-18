import numpy as np
import sys
from scipy.signal import correlate as cc
sys.path.append('../')
import util

def misfit_function_ij2(sig_data, sig_sim, tspace, tmin=None, tmax=None, mode='Waveform'):
    dt = abs(tspace[1] - tspace[0])
    nSamples = len(tspace)

    # Set Cuts
    if tmin != None and tmax != None:
        ii_min = util.findNearest(tspace, tmin)
        ii_max = util.findNearest(tspace, tmax)
    elif tmin != None and tmax == None:
        ii_min = util.findNearest(tspace, tmin)
        ii_max = nSamples
    elif tmin == None and tmax != None:
        ii_min = 1
        ii_max = util.findNearest(tspace, tmax)
    else:
        ii_min = 1
        ii_max = nSamples

    sig_cc = cc(sig_sim, sig_data)
    sig_cc_abs = abs(sig_cc)
    sig_cc_max = max(sig_cc_abs)

def misfit_function_ij(sig_data, sig_sim, tspace, tmin=None, tmax=None, mode='Envelope'):
    dt = abs(tspace[1] - tspace[0])
    nSamples = len(tspace)

    # Set Cuts
    if tmin != None and tmax != None:
        ii_min = util.findNearest(tspace, tmin)
        ii_max = util.findNearest(tspace, tmax)
    elif tmin != None and tmax == None:
        ii_min = util.findNearest(tspace, tmin)
        ii_max = nSamples
    elif tmin == None and tmax != None:
        ii_min = 1
        ii_max = util.findNearest(tspace, tmax)
    else:
        ii_min = 1
        ii_max = nSamples


    #Calculate Misfit
    if mode == 'Waveform':
        chi_local = 0
        for i in range(ii_min,ii_max):
            chi_local += abs(sig_data[i] - sig_sim[i])**2 * dt
    elif mode == 'Envelope':
        chi_local = 0
        for i in range(ii_min, ii_max):
            envelope_sim_i = np.sqrt(sig_sim[i].real**2 +sig_sim[i].imag**2)
            envelope_data_i = np.sqrt(sig_data[i].real**2 + sig_data[i].imag**2)
            chi_local_i = ((np.log(envelope_data_i/envelope_sim_i))**2 ) * dt
            chi_local += chi_local_i
    return chi_local

def misfit_function_ij0(sig_data, sig_sim, tspace, tmin=None, tmax=None, mode='Waveform'):
    dt = abs(tspace[1] - tspace[0])
    nSamples = len(tspace)

    # Set Cuts
    if tmin != None and tmax != None:
        ii_min = util.findNearest(tspace, tmin)
        ii_max = util.findNearest(tspace, tmax)
        nCut = ii_max - ii_min

        t_out = np.linspace(tmin, tmax, nCut)

    elif tmin != None and tmax == None:
        ii_min = util.findNearest(tspace, tmin)
        ii_max = nSamples
        nCut = ii_max - ii_min

        t_out = np.linspace(tmin, max(tspace), nCut)

    elif tmin == None and tmax != None:
        ii_min = 1
        ii_max = util.findNearest(tspace, tmax)
        nCut = ii_max - ii_min

        t_out = np.linspace(min(tspace),tmax, nCut)
    else:
        ii_min = 1
        ii_max = nSamples
        nCut = ii_max - ii_min

        t_out = np.linspace(min(tspace),max(tspace), nCut)


    #Calculate Misfit
    chi_out = np.zeros(nCut)
    chi_sum = np.zeros(nCut)
    if mode == 'Waveform':
        chi_local = 0
        j = 0
        for i in range(ii_min,ii_max):
            #chi_local_i = (abs(np.angle(sig_data[i]) - np.angle(sig_sim[i]))**2) * dt
            envelope_sim_i = np.sqrt(sig_sim[i].real**2 +sig_sim[i].imag**2)
            envelope_data_i = np.sqrt(sig_data[i].real**2 + sig_data[i].imag**2)
            chi_local_i = ((np.log(envelope_data_i/envelope_sim_i))**2 ) * dt
            #chi_local_i = abs(sig_data[i] - sig_sim[i])**2 * dt
            chi_local += chi_local_i
            chi_out[j] = chi_local_i
            chi_sum[j] = chi_local
            j+=1
            #chi_local += abs(sig_data[i] - sig_sim[i])**2 * dt
    return chi_local, chi_out, t_out, chi_sum


def misfit_function(bscan_data, bscan_sim, tspace, tmin=None, tmax=None, mode='Waveform'): #TODO: Write a loop for this over nTX and nRX
    nTX = len(bscan_data)
    nRX = len(bscan_data[0])

    Chi = 0
    for i in range(nTX):
        for j in range(nRX):
            sig_sim = bscan_sim[i,j]
            sig_data = bscan_data[i,j]
            Chi += misfit_function_ij(sig_data=sig_data, sig_sim=sig_sim, tspace=tspace, tmin=tmin, tmax=tmax, mode=mode)
    Chi /= (2. * float(nTX)*float(nRX))
    return Chi

def fitness_function(bscan_data, bscan_sim, tspace, tmin=None, tmax=None, mode='Waveform'):
    Chi = misfit_function(bscan_data, bscan_sim, tspace, tmin, tmax, mode)
    S = 1 / Chi
    return S