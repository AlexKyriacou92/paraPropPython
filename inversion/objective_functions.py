import numpy as np
import sys
from scipy.signal import correlate as cc
sys.path.append('../')
import util
import peakutils as pku
import scipy
def check_symmetric(a, tol=1e-10):
    return np.all((a - np.flip(a)) < tol)

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
    chi_local = 0
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
    elif mode == 'Correlation':
        chi_local = 0
        j = 0
        sig_cross_correl = abs(cc(sig_sim, sig_data))
        sig_auto = abs(cc(sig_data, sig_data))
        t_space_lag = np.linspace(-max(tspace), max(tspace), len(sig_cross_correl))
        indexes = pku.indexes(sig_cross_correl, thres=0.05)
        indexes2 = pku.indexes(sig_auto, thres=0.1)
        nPeaks = len(indexes)
        '''
        print(len(indexes), len(indexes2))
        i_mid = np.argmin(abs(t_space_lag))
        print(len(sig_cross_correl), max(t_space_lag), min(t_space_lag), t_space_lag[i_mid])
        print(len(t_space_lag[i_mid+1:len(t_space_lag)]), len(t_space_lag[0:i_mid]))
        t_positive = t_space_lag[i_mid+1:len(t_space_lag)]
        t_negative = np.flip(t_space_lag[0:i_mid])
        sig_cross_pos = sig_cross_correl[i_mid+1:len(t_space_lag)]
        sig_cross_neg = np.flip(sig_cross_correl[0:i_mid])
        sig_cross_delta = abs(sig_cross_pos / sig_cross_neg)
        peaks_3 = pku.indexes(sig_cross_delta, thres=0.05)
        nPeaks_3 = len(peaks_3)
        '''
        print('Check Array Symmetry')
        #array_symmetry = scipy.linalg.issymmetric(sig_cross_correl)
        #if array_symmetry == False:
        if check_symmetric(sig_cross_correl) == False:
            print('False')
            for j in range(nPeaks):
                k_peaks = indexes[j]
                amp_off = (np.log(sig_cross_correl[k_peaks]/max(sig_auto)))**2
                t_offset = abs(t_space_lag[k_peaks]/dt)
                weight = sig_cross_correl[k_peaks]/sig_auto[k_peaks]

                chi_local += t_offset * weight
                '''
                k_peak = peaks_3[j]
                t_offset = t_positive[k_peak]
                weight = sig_cross_delta[k_peak]
                chi_local += t_offset * weight
                #chi_local += amp_off
                '''
            #i_mid = int((len(t_space_lag) - 1) / 2)
            #chi_local += (sig_cross_correl[i_mid] / sig_auto[i_mid]) * dt
            chi_local += 1
        else:
            print('True')
            i_mid = int((len(t_space_lag) - 1)/2)
            chi_local = (np.log(sig_cross_correl[i_mid]/sig_auto[i_mid]))**2
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
    if mode == 'Envelope':
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
    elif mode == 'Waveform':
        chi_local = 0
        j = 0
        for i in range(ii_min,ii_max):
            chi_local_i = abs(abs(sig_data[i]) - abs(sig_sim[i]))**2 * dt
            chi_local += chi_local_i
            chi_out[j] = chi_local_i
            chi_sum[j] = chi_local
            j += 1
    '''
    elif mode == 'Correlation':
        chi_local = 0
        j = 0
        sig_cross_correl = abs(cc(sig_sim, sig_data))
        t_space_lag = np.linspace(-max(tspace), max(tspace), len(sig_cross_correl))
        k_cut = util.findNearest(t_space_lag, 0)
        t_space_lag = t_space_lag[k_cut:]
        sig_cross_correl = sig_cross_correl[k_cut:]
        k_max = np.argmax(sig_cross_correl)
        t_max_cc = t_space_lag[k_max]
        for i in range(ii_min, ii_max):
            chi_local = abs(sig_data[i].real * sig_sim[i].real)**2 * dt
            chi_local
    '''


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