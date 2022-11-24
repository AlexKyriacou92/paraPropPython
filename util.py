# paraPropPython
# s. prohira
# GPL v3

import matplotlib.pyplot as plt
import numpy as np
import scipy as sci
import scipy.constants as constant
import scipy.io.wavfile as wav
import scipy.signal as sig
import scipy.interpolate as interp
from scipy.signal import butter, lfilter
from numpy import linalg as la
import csv
from numpy.lib.format import open_memmap
import h5py

I=1.j
c_light = .29979246;#m/ns
c0 = c_light*1e9

pi = 3.14159265358979323846; #radians
twoPi = 2.*pi; #radians
z_0=50; #ohms
deg=pi/180.; #radians
kB=8.617343e-11;#MeV/kelvin
kBJoulesKelvin=1.38e-23;#J/kelvin
rho=1.168e-3;#sea level density
x_0=36.7;#radiation length in air
e_0=.078;#ionization energy 

  
m = 1.;
ft = .3047*m;
cm = .01*m;
mm = .001*m;



ns = 1.;
us = ns*1e3;
ms = ns*1e6;
s = ns*1e9;


GHz = 1.;
MHz = .001*GHz;
kHz = 1e-6*GHz;
Hz = 1e-9*GHz;

def lowpassFilter(dt, cutoff, invec):
    period=dt
    w= cutoff*2.*np.pi;
    T = period;
    a = w*T;
    b = np.exp(-w*T);
    out=np.zeros(len(invec))
    
    for i in range(1,len(invec)):
        value = a*invec[i]+b*out[i-1]
        out[i]=value
    return out

def normToMax(a):
    avec=np.zeros(len(a), dtype=type(a))
    avec=a/(np.amax(a))
    return avec


def normalize(a):
    length=len(a)
    avec=np.array(a, dtype='float')
    norm=np.sqrt(np.sum(avec*avec))
    return avec/norm

def normalizeAndSubtract(a, b):
    out=subtract(normalize(a), normalize(b))
    return out

def rms(inArray):
    val=0.
    for i in range(inArray.size):
        val+=inArray[i]*inArray[i]
    return np.sqrt(val/float(inArray.size))

#this is slow and dumb
def getIndex(inX, t):
    for i in range(inX.size):
        if inX[i] > t:
            return i

#stolen from stack overflow lol
def findNearest(array, value):
    array = np.asarray(array, dtype='complex')
    idx = (np.abs(array - value)).argmin()
    return idx#array[idx]


def align(v1, v2):
    test = sig.correlate(v1, v2);


    maxx = np.argmax(test);
    diff=maxx-len(v2)
    
    append = True if diff > 0 else False

    if append is False:
        mask=np.ones(len(v2), dtype=bool)
        mask[0:abs(diff)]=False
        v3=v2[mask, ...]
        v4=np.pad(v3, pad_width=(0, abs(diff)))
        
        return v4

    else:
        mask=np.ones(len(v2), dtype=bool)
        mask[len(mask)-abs(diff):len(mask)]=False
        v3=v2[mask, ...]                     
        v4=np.pad(v3, pad_width=(abs(diff), 0))
        return v4
    # #print append, diff
    
    # if append is False:
    #     zero                  = np.zeros(np.abs(diff-len(v2))+1)
    #     #print len(zero)
    #     #print v2.shape
    #     v3                    = np.insert(v2, 0, zero)
    #     mask                  = np.ones(len(v3), dtype=bool)
    #     mask[len(v1):len(v3)] = False
    #     v4                    = v3[mask,...]
    #     #print v4.shape
    #     return v4
    
    # if append is True:
    #     mask                          = np.ones(len(v2), dtype=bool)
    #     #print len(zero)
    #     mask[:np.abs(diff-len(v2))-1] = False
    #     zero                          = np.zeros(np.abs(diff-len(v2))-1)
        
    #     #print v2.shape
    #     v3  = v2[mask,...]
    #     v4  = np.insert(v3, len(v3)-1, zero)
    #     #v3 = v2
    #     #print v4.shape
    #     return v4


def delayGraph(v1, delay):
    numzeros=int(np.abs(delay))
    zeros=np.zeros(numzeros);
    out=np.zeros(numzeros+len(v1));
    if delay>=0:
        out=np.insert(v1, 0, zeros);

    if delay<0:
        out=np.insert(v1, len(v1)-1, zeros)
    return out

def delayGraphXY(vx, vy, delay):
    dx=vx[1]-vx[0];
    numzeros=int(np.abs(delay)/dx)
    zerosy=np.zeros(numzeros);
    outx=np.zeros(numzeros+len(vy));
    outy=np.zeros(numzeros+len(vy));

    if delay>=0:
        zerosx=np.linspace(0,delay, int(dx*numzeros))
        outy=np.insert(vy, 0, zerosy);
        outx=np.insert(vx, 0, zerosx);
    if delay<0:
        mask=np.ones(len(vy), dtype=bool)
        mask[0:numzeros]=False
        zerosx=np.linspace(-delay, 0, int(dx*numzeros))
        vtemp=vy[mask,...]
        outy=np.insert(vtemp, len(vtemp)-1, zerosy)
        outx=np.insert(vx, len(vx)-1, zerosx)
    return outx, outy

def sampledCW(freq, amp, times, phase):
    values=amp*np.sin(2.*np.pi*freq*times +phase)
    return values

def getPhase(ingr):
    fftGr=doFFT(ingr)
    
    vals=np.arctan2(fftGr.imag, fftGr.real)
   
    return vals


def getFresnelR(n1, n2, angleDeg, pol=0):
    angleRad=np.deg2rad(angleDeg);
    theta=np.arcsin(n1*np.sin(angleRad)/n2)
    th = (2.*n1*np.cos(angleRad))/(n1*np.cos(angleRad)+n2*np.cos(theta));
    tv = (2.*n1*np.cos(angleRad))/(n1*np.cos(theta)+n2*np.cos(angleRad));
    Th = (n2*np.cos(theta)*th*th)/(n1*np.cos(angleRad))
    Tv = (n2*np.cos(theta)*tv*tv)/(n1*np.cos(angleRad));

    if (pol==0): return Th
    else: return Tv
    
def makeCW(freq, amp, t_min, t_max, GSs, phase):

    dt=1./GSs
    tVec=np.arange(t_min, t_max, dt);
    N=tVec.size
    outx=np.zeros(N);
    outy=np.zeros(N);
    index=0
    for t in tVec:
        temp=amp*np.sin(2.*np.pi*freq*t +phase)
        outy[index]=temp;
        outx[index]=t
        index+=1;
    return outx, outy

def power(V, start, end):
    powV=V*V
    return np.sum(powV[start:end])

def doFFT(V):
    return np.fft.fft(V)

def doIFFT(V):
    return np.fft.ifft(V)

def hilbertTransform(V):
    return np.imag(sig.hilbert(V));


# ff=doFFT(V);
    # for i in range(len(ff)/4):
    #     temp=ff.imag[i]
    #     ff.imag[i]=ff.real[i]
    #     ff.real[i]=-1.*temp
    # outf=doIFFT(ff)
    # return np.array(outf)

def hilbertEnvelope(V):
    h=hilbertTransform(V)
    return np.array(np.sqrt(V*V+h*h)).real

def interpolate(data, factor):
    x=np.linspace(0,len(data)-1, len(data));
#    #print len(x), len(data)
    tck = interp.splrep(x, data, s=0)
    xnew = np.linspace(0,len(data)-1, len(data)*factor)
    ynew = interp.splev(xnew, tck, der=0)
#    #print len(ynew)
    return ynew

def interpolate2D(data, factorx, factory=1):
    y=np.linspace(0,len(data[0])-1, len(data[0]));
    x=np.linspace(0,len(data)-1, len(data));
#    #print len(x), len(data)
    spline = interp.RectBivariateSpline(x,y,data)
    xnew = np.linspace(0,len(x), len(x)*factorx)
    ynew =np.linspace(0,len(y), len(y)*factory)
    out=spline(xnew, ynew)
#    #print len(ynew)
    return out


def sincInterpolate(datax, datay, GSs):
    T=datax[1]-datax[0]
    dt=1./GSs    
    tVec=np.arange(0., datax[datax.size-1], dt);
    nPoints=tVec.size
    outx=np.zeros(tVec.size)
#    print "sz", outx.size
    outy=np.zeros(tVec.size)
    outx=np.zeros(nPoints)
    #print outx.size
    outy=np.zeros(nPoints)
    t=0.
    index=0;
    ind=np.arange(0, datay.size, 1)
    for t in tVec:
        temp=0;
        sVec=datay*np.sinc((t-ind*T)/T)
        for i in range(len(datay)):
           # temp+=datay[i]*np.sinc((t-(float(i)*T))/T);
            temp+=sVec[i];
        outy[index]=temp;
        outx[index]=t

        index+=1
    return outx, outy

def sincInterpolateFast(datax, datay, GSs, N=10):
    T=datax[1]-datax[0]
    dt=1./GSs
    tVec=np.arange(0., datax[datax.size-1], dt);
    outx=np.zeros(tVec.size)
    #print "sz", outx.size
    outy=np.zeros(tVec.size)
    t=0.
    index=0;
    ind=np.arange(0., datay.size, 1.)
    for t in tVec:
        temp=0;
        smallIndex=int(t/T);
        ilow=smallIndex-N;
        ihigh=smallIndex+N;
        if(ilow<0):
            ilow=0
        if(ihigh>=datay.size):
            ihigh=datay.size-1
        sVec=datay[ilow:ihigh]*np.sinc((t-ind[ilow:ihigh]*T)/T)
#        print sVec.size
        for i in range(0,ihigh-ilow):
            #temp+=datay[i]*np.sinc((t-(float(i)*T))/T);
            temp+=sVec[i]
        outy[index]=temp;
        outx[index]=t
        index+=1
#        print (ilow, " ", ihigh)
    return outx, outy


def butterBandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butterBandpassFilter(data, lowcut, highcut, fs, order=3):
    b, a = butterBandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def butterHighpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = sig.butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def butterHighpassFilter(data, cutoff, fs, order=5):
    b, a = butterHighpass(cutoff, fs, order=order)
    y = sig.filtfilt(b, a, data)
    return y

def butterLowpass(highcut, fs, order=5):
    nyq = 0.5 * fs
    high = highcut / nyq
    b, a = butter(order, high, btype='low')
    return b, a


def butterLowpassFilter(data, highcut, fs, order=3):
    b, a = butterLowpass(highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


def getZeroCross(datax, datay):
    signDat = (datay > 0).astype(int)
    offsetDat=np.roll(signDat, 1)
    vec=np.logical_xor(signDat, offsetDat)
    tVec=np.ediff1d(np.trim_zeros(np.sort(vec*datax)));
    return tVec

def dot(one, two, norm=1):
    prod=np.dot(one, two)
    denom=np.sqrt(np.dot(one, one) * np.dot(two, two))
    if norm==1:
        out=prod/denom
    else:
        out=prod
    return out

def reflection_coefficient(m1, m2):
    n1 = m1.real
    k1 = m1.imag
    n2 = m2.real
    k2 = m2.imag
    #return abs(n1-n2)/abs(n1+n2)

    return np.sqrt((abs(n1 - n2)**2 + abs(k1-k2)**2) / (abs(n1+n2)**2 + abs(k1 + k2)**2))

#Transmission Coefficient
def transmission_coefficient(m1, m2):
    return 1 - reflection_coefficient(m1, m2)

def create_memmap(file, dimensions, data_type ='complex'):
    A = open_memmap(file, shape = dimensions, mode='w+', dtype = data_type)
    return A

def get_maximum_index(arr):
    ii = np.argmax(arr)
    return ii

def get_maximum_x(x, y):
    ii = np.argmax(y)
    xMax = x[ii]
    return xMax

def roll_back(t, sig, t_expected):
    ii_sim = get_maximum_index(abs(sig))
    ii_expected = findNearest(t, t_expected)
    if ii_sim >= ii_expected:
        delta_ii = ii_sim - ii_expected
    elif ii_sim < ii_expected:
        delta_ii = ii_expected - ii_sim
    sig_roll = np.roll(sig, -delta_ii)
    return sig_roll

#=================================================================================================
'''
Analysis Tools
'''
#=================================================================================================

def gaussian(x, mu, sig, norm, base):  # gaussian distribution + nooise
    a = -1. * pow(x - mu, 2.) / (2 * pow(sig, 2.))
    return norm * np.exp(a) + base


def gaussian_simple(x, mu, sig, norm):  # gaussian distribution + nooise
    a = -1. * pow(x - mu, 2.) / (2 * pow(sig, 2.))
    return norm * np.exp(a)

def cut_waveform(wvf, dt, t1, t2):
    N = len(wvf)
    tmax = N*dt
    n1 = int(t1/tmax * N)
    n2 = int(t2/tmax * N)
    tspace = np.linspace(0, tmax, N)
    return wvf[n1:n2], tspace[n1:n2]

def median_power(spectrum, dt, f1, f2):
    N = len(spectrum)
    f_nyq = (1/(2*dt))
    print('f_nyq = ', f_nyq)
    n1 = int(f1/f_nyq * float(N))
    n2 = int(f2/f_nyq * float(N))
    print(n1, n2, n2-n1)
    spectrum_cut = spectrum[n1:n2]
    print('spectrum cut =', spectrum_cut)
    print(np.median(spectrum_cut))
    return np.median(spectrum_cut)

def padd_wvf(arr, index):
    nArr = len(arr)
    hArr = int(float(len(arr)) / 2)
    nPadd = 2 ** index
    padded_wvf = np.zeros(nPadd)
    hPadd = int(float(nPadd) / 2)
    nPadd_cut = len(padded_wvf[hPadd - hArr: hPadd + hArr])
    if nPadd_cut == nArr:
        padded_wvf[hPadd - hArr: hPadd + hArr] = arr
    elif nPadd_cut > nArr:
        dArr = nPadd_cut - nArr
        padded_wvf[hPadd - hArr: hPadd + hArr - dArr] = arr
    elif nArr > nPadd_cut:
        padded_wvf[hPadd - hArr: hPadd + hArr] = arr[:nPadd_cut]
    return padded_wvf

def padded_frequency(index, dt):
    nPadd = 2**index
    return np.fft.rfftfreq(nPadd, dt)

fitfunc = lambda p, x: gaussian(x, p[0], p[1], p[2], p[3])  # fit function for SciPy optimize
errfunc = lambda p, x, y: fitfunc(p, x) - y  # residual function -> minimized by SciPy optimize

fitfunc_simple = lambda p, x: gaussian_simple(x, p[0], p[1], p[2])  # fit function for SciPy optimize
errfunc_simple = lambda p, x, y: fitfunc_simple(p, x) - y  # residual function -> minimized by SciPy optimize

def get_fft_data(fname_hdf, t_cut0 = 0, t_cut1 = 600e-9):
    hdf_input = h5py.File(fname_hdf, 'r')
    wvf_ave = np.array(hdf_input.get('wvf_ave'))
    dt = float(hdf_input.attrs['dt'])
    nSamples = int(hdf_input.attrs['nSamples'])
    tspace = np.linspace(0, nSamples * dt, nSamples)

    B = float(hdf_input.attrs['band'])
    dfdt = float(hdf_input.attrs['dfdt'])
    T = float(hdf_input.attrs['T'])
    Tmin = 0.05 * T
    Tmax = 0.95 * T

    L = float(hdf_input.attrs['L_tx']) - float(hdf_input.attrs['L_rx'])
    R = float(hdf_input.attrs['R'])

    t_cutoff = L/(c0*0.66)
    fcutoff = dfdt * L / (c0 * 0.66) / 4.

    # Procedure -->
    # Cutt Wvf --->
    wvf_cut, t_cut = cut_waveform(wvf_ave, dt, Tmin, Tmax)

    # Apply High Pass Filter > dfdt * L/v_cable / 4 -> 2.5 kHz
    wvf_filt = butterHighpassFilter(wvf_cut, fcutoff, 1 / dt, order=3)

    # Apply Blackman Window Function
    wvf_filt *= np.blackman(len(wvf_filt))

    # Apply Padding
    i_padd = 20
    wvf_padd = padd_wvf(wvf_filt, i_padd)

    fft_padd = np.fft.rfft(wvf_padd)
    freq_padd = padded_frequency(i_padd, dt)

    fft_abs = abs(fft_padd) ** 2
    time_of_flight = freq_padd / dfdt - t_cutoff
    ii_min = findNearest(time_of_flight, t_cut0)
    ii_max = findNearest(time_of_flight, t_cut1)
    fft_abs_out = fft_abs[ii_min:ii_max]
    time_of_flight_out = time_of_flight[ii_min:ii_max]
    hdf_input.close()
    return fft_abs_out, time_of_flight_out

def get_waveform_simulation(fname, zTx, xRx, zRx):
    fname_h5 = fname + '.h5'
    fname_npy = fname + '.npy'

    input_hdf = h5py.File(fname_h5, 'r')

    dt = input_hdf.attrs["dt"]
    nSamples = input_hdf.attrs["nSamples"]

    tspace = np.linspace(0, dt * nSamples, nSamples) * 1e-9
    bscan = np.load(fname_npy, 'r')

    tx_depths = np.array(input_hdf.get('source_depths'))
    rx_depths = np.array(input_hdf.get('rx_depths'))
    rx_ranges = np.array(input_hdf.get('rx_range'))

    ii_tx_z = findNearest(tx_depths, zTx)
    ii_rx_x = findNearest(rx_ranges, xRx)
    ii_rx_z = findNearest(rx_depths, zRx)

    sig_rx = bscan[ii_tx_z, ii_rx_x, ii_rx_z]
    return sig_rx, tspace

def create_hdf(fname, sim, tx_signal, tx_depths, rx_ranges, rx_depths, comment=""):
    output_hdf = h5py.File(fname, 'w')
    output_hdf.attrs["iceDepth"] = sim.iceDepth
    output_hdf.attrs["iceLength"] = sim.iceLength
    output_hdf.attrs["airHeight"] = sim.airHeight
    output_hdf.attrs["dx"] = sim.dx
    output_hdf.attrs["dz"] = sim.dz
    output_hdf.create_dataset('n_matrix', data=sim.get_n())

    output_hdf.attrs["Amplitude"] = tx_signal.amplitude
    output_hdf.attrs["freqCentral"] = tx_signal.frequency
    output_hdf.attrs["Bandwidth"] = tx_signal.bandwidth
    output_hdf.attrs["freqLP"] = tx_signal.freqMax
    output_hdf.attrs["freqHP"] = tx_signal.freqMin
    output_hdf.attrs["freqSample"] = tx_signal.fsample
    output_hdf.attrs["freqNyquist"] = tx_signal.freq_nyq
    output_hdf.attrs["tCentral"] = tx_signal.t_centre
    output_hdf.attrs["tSample"] = tx_signal.tmax
    output_hdf.attrs["dt"] = tx_signal.dt
    output_hdf.attrs["nSamples"] = tx_signal.nSamples

    n_profile_data = np.zeros((2, len(sim.get_n(x=0))))
    n_profile_data[0] = sim.z
    n_profile_data[1] = sim.get_n(x=0)

    nRX_x = len(rx_ranges)
    nRX_z = len(rx_depths)
    rxArray = np.ones((nRX_x, nRX_z, 2))
    for i in range(nRX_x):
        for j in range(nRX_z):
            rxArray[i,j,0] = rx_ranges[i]
            rxArray[i,j,1] = rx_depths[j]

    output_hdf.create_dataset("rxArray", data=rxArray)
    output_hdf.create_dataset('n_profile', data=n_profile_data)
    output_hdf.create_dataset("source_depths", data=tx_depths)
    output_hdf.create_dataset('tspace', data=tx_signal.tspace)
    output_hdf.create_dataset('signalPulse', data=tx_signal.pulse)
    output_hdf.create_dataset('signalSpectrum', data=tx_signal.spectrum)
    output_hdf.create_dataset("rx_range", data= rx_ranges)
    output_hdf.create_dataset("rx_depths", data = rx_depths)

    output_hdf.attrs["comment"] = comment

    return output_hdf