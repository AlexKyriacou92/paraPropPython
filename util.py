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

I=1.j
c_light = .29979246;#m/ns
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

def sincInterpolateFast(datax, datay, GSs, N=10): #TODO: Note to Cade and Steven: some comments would be nice! What is GSs??
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

def make_pulse(t, t_central, amplitude, freq_central):
    pulse = amplitude * sig.gausspulse(t - t_central, freq_central)
    return pulse

def create_memmap(file, dimensions, data_type ='complex'):
    A = open_memmap(file, shape = dimensions, mode='w+', dtype = data_type)
    return A

def print_var_name(variable):
    label = ''
    for name in globals():
        if eval(name) is variable:
            print(name)
            label = name
    return label

#Functions -> to read input functions

def select_variable(var_str,fname):
    output_val = np.nan
    with open(fname) as f:
        for line in f:
            columns = line.split()
            if len(columns) > 0:
                label = str(columns[0])
                filevar_str = label[1:]
                if var_str == filevar_str:
                    value = float(columns[1])
                    output_val = value
    return output_val

def select_string(var_str,fname):
    output_str = ''
    with open(fname) as f:
        for line in f:
            columns = line.split()
            if len(columns) > 0:
                label = str(columns[0])
                filevar_str = label[1:]
                if var_str == filevar_str:
                    string = str(columns[1])
                    output_str = string
    return output_str

def count_objects(obj_type, fname):
    nObjs = 0
    with open(fname) as f:
        for line in f:
            columns = line.split()
            if len(columns) > 0:
                label = str(columns[0])
                filevar_str = label[1:]
                if obj_type == filevar_str:
                    nObjs += 1
    return nObjs

from geometry import triangle, circle
from shapely.geometry import Point
from permittivity import *


def select_crevass(fname):
    crevass_list = []
    with open(fname) as f:
        for line in f:
            columns = line.split()
            if len(columns) > 0:
                label = str(columns[0])
                filevar_str = label[1:]
                if filevar_str == "crevass":
                    x1 = float(columns[1])
                    z1 = float(columns[2])
                    p1 = Point(x1,z1)

                    x2 = float(columns[3])
                    z2 = float(columns[4])
                    p2 = Point(x2,z2)

                    x3 = float(columns[5])
                    z3 = float(columns[6])
                    p3 = Point(x3,z3)

                    n_object = 1+0j
                    triangle_i = triangle(p1,p2,p3, n_object)
                    crevass_list.append(triangle_i)
    return crevass_list

def select_aquifer(fname):
    aquifer_list = []
    with open(fname) as f:
        for line in f:
            columns = line.split()
            if len(columns) > 0:
                label = str(columns[0])
                filevar_str = label[1:]
                if filevar_str == "aquifer":
                    x1 = float(columns[1])
                    z1 = float(columns[2])
                    p1 = Point(x1,z1)

                    x2 = float(columns[3])
                    z2 = float(columns[4])
                    p2 = Point(x2,z2)

                    x3 = float(columns[5])
                    z3 = float(columns[6])
                    p3 = Point(x3,z3)

                    n_object = eps2m(eps_water)
                    triangle_i = triangle(p1,p2,p3, n_object)
                    aquifer_list.append(triangle_i)
    return aquifer_list

def select_triangle(fname):
    triangle_list = []
    with open(fname) as f:
        for line in f:
            columns = line.split()
            if len(columns) > 0:
                label = str(columns[0])
                filevar_str = label[1:]
                if filevar_str == "triangle":
                    x1 = float(columns[1])
                    z1 = float(columns[2])
                    p1 = Point(x1,z1)

                    x2 = float(columns[3])
                    z2 = float(columns[4])
                    p2 = Point(x2,z2)

                    x3 = float(columns[5])
                    z3 = float(columns[6])
                    p3 = Point(x3,z3)

                    n_object = 1+0j
                    triangle_i = triangle(p1,p2,p3, n_object)
                    triangle_list.append(triangle_i)
    return triangle_list

def select_circle(fname):
    circle_list = []
    with open(fname) as f:
        for line in f:
            columns = line.split()
            if len(columns) > 0:
                label = str(columns[0])
                filevar_str = label[1:]
                if filevar_str == "circle":
                    x = float(columns[1])
                    z = float(columns[2])
                    r = float(columns[3])
                    n_circle = float(columns[3])

                    circle_i = circle(x,z,r, n_circle)
                    circle_list.append(circle_i)
    return circle_list

def select_meteor(fname):
    meteor_list = []
    with open(fname) as f:
        for line in f:
            columns = line.split()
            if len(columns) > 0:
                label = str(columns[0])
                filevar_str = label[1:]
                if filevar_str == "meteor":
                    x = float(columns[1])
                    z = float(columns[2])
                    r = float(columns[3])
                    n_meteor = eps2m(eps_meteor)
                    circle_i = circle(x,z,r, n_meteor)
                    meteor_list.append(circle_i)
    return meteor_list


def select_range(var_str,fname):
    output_array = np.nan
    with open(fname) as f:
        for line in f:
            columns = line.split()
            if len(columns) > 0:
                label = str(columns[0])
                filevar_str = label[1:]
                if var_str == filevar_str:
                    min_val = float(columns[1])
                    max_val = float(columns[2])
                    val_step = float(columns[3])
                    output_array = np.arange(min_val, max_val, val_step)
    return output_array

#Implement Backwards Solution -> based on do_solver()
def cut_xaxis(x, xmin, xmax):
    idx_min = findNearest(xmin, x)
    idx_max = findNearest(xmax, x)
    return x[idx_min:idx_max]

#Reflection Coefficient -> inputs can be complex
def reflection_coefficient(m1, m2):
    if any(m1.imag) > 0 or any(m2.imag) > 0:
        n1 = m1.real
        k1 = m1.imag
        n2 = m2.real
        k2 = m2.imag
        return (abs(n1 - n2)**2 + abs(k1-k2)**2) / (abs(n1+n2)**2 + abs(k1 + k2)**2)
    else:
        n1 = m1.real
        n2 = m2.real
        return (abs(n1 - n2)**2) / (abs(n1+n2)**2)

    #return abs(n1.real - n2.real) / (n1.real + n2.real)

#Transmission Coefficient
def transmission_coefficient(m1, m2):
    return 1 - reflection_coefficient(m1, m2)