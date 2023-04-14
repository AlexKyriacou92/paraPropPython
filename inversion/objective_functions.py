import numpy as np
import sys
sys.path.append('../')
import util

def misfit_function_ij(sig_data, sig_sim, tspace, tmin=None, tmax=None, mode='Waveform'):
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
    return chi_local

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