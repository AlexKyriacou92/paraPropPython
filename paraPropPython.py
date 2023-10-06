# paraPropPython
# c. sbrocco, s. prohira

import util
import math as m
from math import pi
import numpy as np
from inspect import signature
from scipy.interpolate import interp1d
from scipy import signal
from receiver import receiver as rx
from transmitter import tx_signal

class paraProp:
    """
    Parameters
    ----------
    iceLength : float
        length of the simulation (m)
    iceDepth : float
        depth of the ice simulated (m)
    dx : float
        grid spacing in the x direction (m)
    dz : float
        grid spacing in the z direction (m)
    airHeight : float
        amount of air to be simulated above ice (m). Initialized to 25 m
    filterDepth : float
        size of the filtered reason above and below simulated region (m). Initialized to 100 m
    refDepth : float
        reference depth for simulation (m). Initialized to 1 m below surface
    """
    def __init__(self, iceLength, iceDepth, dx, dz, airHeight=25, filterDepth=100, refDepth=1):
        ### spatial parameters ### 
        # x #
        self.x = np.arange(0, iceLength+dx, dx)
        self.xNum = len(self.x)
        self.dx = dx
        
        # z #
        self.iceDepth = iceDepth
        self.airHeight = airHeight
        self.iceLength = iceLength
        self.z = np.arange(-airHeight, iceDepth + dz, dz)
        self.zNum = len(self.z)
        self.dz = dz
        self.refDepth = refDepth            
        
        ### other simulation variables ###       
        # filter information #
        self.fNum0 = int(filterDepth / dz)
        
        self.fNum1, self.fNum2 = self.optimize_filt_size(self.zNum, self.fNum0)
        self.zFull = np.arange(-(airHeight + self.fNum1*dz), iceDepth + self.fNum2*dz + dz, dz)
        self.zNumFull = len(self.zFull)
        win = np.blackman(self.fNum1 + self.fNum2)
        filt = np.ones(self.zNumFull)
        filt[:self.fNum1] = win[:self.fNum1]
        filt[-self.fNum2:] = win[self.fNum1:]
        self.filt = filt
        #self.Q_option = Q_option

        # z wavenumber # TODO: -> does this wave number have to be adjusted by the refractive index?
        self.kz = np.fft.fftfreq(self.zNumFull)*2*np.pi/self.dz
        
        # index of refraction array #
        # Alex: Change to an array of complex numbers -> to account for attenuation
        self.n = np.ones((self.zNumFull, self.xNum), dtype='complex')
        self.epsilon_r = np.ones((self.zNumFull, self.xNum), dtype='complex')
        #TODO: Should I define it (zNum, xNum) or (xNum, zNum)? Because it's defined (zNum, xNum) in set_n but then is transformed by np.tranpose to (xNum, zNum)

        # source array #
        self.source = np.zeros(self.zNumFull, dtype='complex')
        
        # 2d field array #
        self.field = np.zeros((self.xNum, self.zNum), dtype='complex')

        #2d field array for reflected signals
        self.field_plus = np.zeros((self.xNum, self.zNum), dtype='complex')
        self.field_minus = np.zeros((self.xNum, self.zNum), dtype='complex')

        
    def optimize_filt_size(self, zNum, fNum):
        zNumFull = 2*fNum + zNum
        p2 = 2**m.ceil(m.log(zNumFull, 2))
        p3 = 3**m.ceil(m.log(zNumFull, 3))
        p5 = 5**m.ceil(m.log(zNumFull, 5))
        p7 = 7**m.ceil(m.log(zNumFull, 7))
        p = min([p2, p3, p5, p7])
        fNum = p - zNum
        fNum1 = int(fNum/2)
        fNum2 = fNum - fNum1
        return fNum1, fNum2
        
    def get_x(self):
        """
        gets x grid of simulation
        
        Returns
        -------
        1-d float array
        """
        return self.x
    
    def get_z(self):
        """
        gets z grid of simulation
        
        Returns
        -------
        1-d float array
        """
        return self.z
      
    
    ### ice profile functions ###
    def set_n(self,  nVal=None, nVec=[], nFunc=None, nAir=1.0003, zVec=[], interpolation='padding'):
        """
        Function which sets 2D refractive index profile n(x,z) of simulation domain

        You may define the ref-index profile with a 1D analytical function nFunc,
        or a 1D vector nVec (which may be data defined). For a range independent profile, we take z >= 0 as being
        below the surface (ice) and z < 0 as being above the surface (air/vacuum).
        You can also define a constant profile with nVal
        and the profile will assume uniformity over range x
        Otherwise, you may use a 2D nVec -> where the ref-index as a function of x and z is defined elsewhere and beforehand

        Note on uneven or 'rough' surfaces -> at the moment, this can be implemented with a 2D nVec directly
        Alternatively, you set a 1D ref-index profile as a first step, in the second step you can adjust this profile
        to a changing elevation
         Parameters
        ----------
        nVal : float
            Postcondition: n(z>=0, x>=0) = nVal
        nVec : array
            1-d or 2-d array of float values
            Precondition: spacing between rows is dz, spacing between columns is dx
            Postcondition: n(z=0,x=0) = nVec[0,0], n(z=dz,x=dx) = nVec[1,1], ..., n(z>=len(nVec[:,0])*dz,x>=len(nVec[0,:])*dx) = nVec[-1,-1]
        nFunc : function
            Precondition: nFunc is a function of one or two variables, z and x, and returns a float value
            Postcondition: n(z>=0,x>=0) = nFunc(z,x)
        nAir : float
            index of refraction of air
            Postcondition: n(z<0) = nAir
        zVec : a 1D array of depth positions (needed for data defined ref-index profiles)
        zVec : 1 1D array of range positions
        """
        self.n = np.ones((self.zNumFull, self.xNum), dtype='complex')

        if nVal != None:
            ix_cut = util.findNearest(self.zFull, 0)
            self.n[ix_cut:,:] = nVal
            self.n[:ix_cut, :] = nAir
        elif nFunc != None:
            ix_min = 0
            ix_cut = util.findNearest(self.zFull, 0)
            self.n[:ix_cut, :] = nAir
            sig = signature(nFunc)
            numParams = len(sig.parameters)
            if numParams == 1:
                for i in range(ix_cut, self.zNumFull):
                    if self.zFull[i] <= self.iceDepth:
                        z = self.zFull[i]
                    else:
                        z = self.iceDepth
                    self.n[i, :] = nFunc(z)
            elif numParams == 2:
                for i in range(ix_cut, self.zNumFull):
                    if self.zFull[i] <= self.iceDepth:
                        z = self.zFull[i]
                    else:
                        z = self.iceDepth
                    for j in range(self.xNum):
                        x = self.x[j]
                        self.n[i,j] = nFunc(z,x)
        elif nFunc == None and nVal == None: #Add Smoothing Function
            if len(nVec.shape) == 1:
                ix_cut = util.findNearest(self.zFull, 0)
                ix_min = util.findNearest(self.zFull, min(zVec))
                ix_max = util.findNearest(self.zFull, max(zVec))
                self.n[:ix_cut, :] = nAir

                if len(zVec) != 0:
                    dz_vec = abs(zVec[1] - zVec[0])
                    z_min = min(zVec)
                    z_max = max(zVec)
                    if dz_vec == self.dz:
                        i_zero = util.findNearest(self.zFull, 0)
                        if z_min > 0:
                            print(ix_min, ix_max)
                            for j in range(self.xNum):
                                self.n[i_zero:ix_min,j] = nVec[0]
                                self.n[ix_max:,j] = nVec[-1]
                                for i in range(ix_min, ix_max):
                                    i_vec = util.findNearest(zVec, self.zFull[i])
                                    self.n[i,j] = nVec[i_vec]
                        else:
                            for j in range(self.xNum):
                                self.n[i_zero:, j] = nVec[0]
                                self.n[ix_max:, j] = nVec[-1]
                                for i in range(i_zero, ix_max):
                                    i_vec = util.findNearest(zVec, self.zFull[i])
                                    self.n[i, j] = nVec[i_vec]
                    elif dz_vec > self.dz:
                        zVec_new = np.arange(min(zVec), max(zVec) + self.dz, self.dz)
                        nVec_new = np.ones(len(zVec_new), dtype='complex')
                        if interpolation == 'padding':
                            nVec_new, zVec_2 = util.smooth_padding(z_vec=zVec, n_vec=nVec, dz=self.dz)
                            print(nVec_new)
                            print(zVec_2)
                        else:
                            f_interp = interp1d(zVec, nVec)
                            nVec_new[0] = nVec[0]
                            nVec_new[-1] = nVec[-1]
                            nVec_new[1:-1] = f_interp(zVec_new[1:-1])
                        for i in range(ix_cut, ix_max):
                            z = self.zFull[i]
                            i_vec = util.findNearest(zVec_new, z)
                            self.n[i,:] = nVec_new[i_vec]
                        self.n[ix_max:,:] = nVec_new[-1]
                        self.n[ix_cut:ix_min,:] = nVec_new[0]
                    elif dz_vec < self.dz:
                        zVec_new = np.arange(min(zVec), max(zVec) + self.dz, self.dz)
                        nVec_new = np.ones(len(zVec_new), dtype='complex')
                        f_interp = interp1d(zVec, nVec)
                        nVec_new[0] = nVec[0]
                        nVec_new[-1] = nVec[-1]
                        nVec_new[1:-1] = f_interp(zVec_new[1:-1])
                        for i in range(ix_cut, ix_max):
                            z = self.zFull[i]
                            i_vec = util.findNearest(zVec_new, z)
                            self.n[i,:] = nVec_new[i_vec]
                        self.n[ix_max:, :] = nVec_new[-1]
                        self.n[ix_cut:ix_min, :] = nVec_new[0]
                elif len(nVec) == self.zNumFull:
                    for j in range(self.xNum):
                        self.n[:,j] = nVec
                else:
                    print('error! nVec must have the same length as the simulation domain depth+height')
                    return -1
            elif len(nVec.shape) == 2:
                zNumVec = len(nVec)
                xNumVec = len(nVec[0])
                if zNumVec == self.zNumFull and xNumVec == self.xNum:
                    for i in range(self.zNumFull):
                        for j in range(self.xNum):
                            self.n[i, j] = nVec[i, j]
                else:
                    print('error! the nVec must be set to be equal to be equal to the shape of ref-index domain matrix sim.n')
        else:
            print('error! you must choose between nFunc, nVal and nVec')
            return -1
        ### set reference index of refraction ###
        self.n0 = self.at_depth(self.n[:, 0], self.refDepth)
        self.n = np.transpose(self.n)

    def set_DEM(self, surf_val = None, func_DEM=None, vec_DEM = [], xVec= [], nAir=1.0003, mode = 'shift', interpolation='padding'):
        #NOTE set_n MUST BE SET BEFORE
        nFlat = np.transpose(self.n)
        nNew = np.ones((self.zNumFull, self.xNum))
        self.DEM = np.zeros(self.xNum)
        if surf_val != None:
            i_shift = int(surf_val/self.dz)
            if i_shift > 0:
                if mode == 'shift':
                    for j in range(self.xNum):
                        nNew[:,j] = np.roll(nFlat[:,j], i_shift)
                        nNew[:i_shift,j] = nAir
                elif mode == 'cutting':
                    ix_flat = util.findNearest(self.zFull,0)
                    for j in range(self.xNum):
                        nNew[:i_shift, j] = nAir
                        nNew[i_shift:, j] = nFlat[ix_flat:-i_shift,j]
                else:
                    print('error! mode should be set to: shift or cutting, no effect made')
            elif i_shift < 0:
                if mode == 'shift':
                    for j in range(self.xNum):
                        nNew[:,j] = np.roll(nFlat[:,j], i_shift)
                        nNew[:i_shift, j] = nFlat[-1, j]
                elif mode == 'cutting':
                    i_cut = util.findNearest(self.zFull, surf_val)
                    for j in range(self.xNum):
                        nNew[:, j] = np.roll(nFlat[:, j], i_shift)
                        nNew[:i_cut,j] = nFlat[-1,j]
                else:
                    print('error! mode should be set to: shift or cutting, no effect made')
        elif func_DEM != None:
            if mode == 'shift':
                for j in range(self.xNum):
                    z_shift = func_DEM(self.x[j])
                    self.DEM[j] = z_shift
                    i_shift = int(z_shift/self.dz)
                    nNew[:,j] = np.roll(nFlat[:,j], i_shift)
                    if z_shift > 0:
                        nNew[:i_shift,j] = nAir
                    elif z_shift < 0:
                        nNew[i_shift:,j] = nFlat[-1,j]
            elif mode == 'cutting':
                for j in range(self.xNum):
                    z_shift = func_DEM(self.x[j])
                    self.DEM[j] = z_shift
                    if z_shift > 0:
                        i_cut = util.findNearest(self.zFull, z_shift)
                        nNew[:i_cut, j] = nAir
                        nNew[i_cut:, j] = nFlat[i_cut:, j]
            else:
                print('error! mode should be set to: shift or cutting, no effect made')
        elif func_DEM == None and surf_val == None:
            vecNum = len(vec_DEM)
            if vecNum > 0:
                if vecNum == self.xNum:
                    for j in range(self.xNum):
                        z_shift = vec_DEM[j]
                        self.DEM[j] = z_shift
                        #i_shift_full = util.findNearest(self.zFull, z_shift)
                        i_shift = int(z_shift/self.dz)
                        if mode == 'shift':
                            if i_shift == 0:
                                nNew[:,j] = nFlat[:,j]
                                #print(nNew[:,j])
                            else:
                                nNew[:,j] = np.roll(nFlat[:,j], i_shift)
                                if z_shift > 0:
                                    nNew[:i_shift,j] = nAir
                                elif z_shift < 0:
                                    nNew[i_shift:,j] = nFlat[-1,j]
                                else:
                                    nNew[i_shift,j] = nFlat[-1,j]
                        elif mode == 'cutting':
                            if z_shift > 0:
                                nNew[:i_shift,j] = nAir
                                nNew[i_shift:,j] = nFlat[i_shift:,j]
                else:
                    dx_vec = abs(xVec[1]-xVec[0])
                    xmin = min(xVec)
                    xmax = max(xVec)
                    xVec_new = np.arange(xmin, xmax + self.dx, self.dx)
                    zVec_new = np.zeros(len(xVec_new))
                    if dx_vec > self.dx:
                        if interpolation == 'padding':
                            zVec_new = util.smooth_padding(xVec,vec_DEM, self.dx)
                        else:
                            f_interp = interp1d(xVec, vec_DEM)
                            zVec_new[0] = vec_DEM[0]
                            zVec_new[-1] = vec_DEM[-1]
                            zVec_new[1:-1] = f_interp(xVec_new[1:-1])
                    elif dx_vec < self.dx:
                        f_interp = interp1d(xVec, vec_DEM)
                        zVec_new[0] = vec_DEM[0]
                        zVec_new[-1] = vec_DEM[-1]
                        zVec_new[1:-1] = f_interp(xVec_new[1:-1])
                    j_min = util.findNearest(self.x, xmin)
                    j_max = util.findNearest(self.x, xmax)
                    for j in range(j_min, j_max):
                        j_vec = util.findNearest(xVec_new, self.x[j])
                        self.DEM[j] = zVec_new[j_vec]
                        i_shift = int(self.DEM[j]/self.dz)
                        i_shift_full = util.findNearest(self.zFull, self.DEM[j])
                        if mode == 'shift':
                            nNew[:,j] = np.roll(nFlat[:,j], i_shift)
                            if self.DEM[j] > 0:
                                nNew[:i_shift, j] = nAir
                            elif self.DEM[j] < 0:
                                nNew[i_shift:, j] = nFlat[-1,j]
                        elif mode == 'cutting':
                            if self.DEM[j] > 0:
                                nNew[:i_shift_full,j] = nAir
                                nNew[i_shift_full:,j] = nFlat[i_shift_full:,j]
            else:
                print('error! DEM vector must be set if function and surface_shift are set to none')
        self.n = np.transpose(nNew)

    def get_n(self, x=None, z=None):
            """
            gets index of refraction profile of simulation

            Returns
            -------
            2-d float array
            """
            if x == None and z == None:
                return np.transpose(self.n[:,self.fNum1:-self.fNum2])
            elif x == None and z != None:
                ii = util.findNearest(self.zFull, z)
                return self.n[:,ii]
            elif z == None and x != None:
                ii = util.findNearest(self.x, x)
                return self.n[ii,self.fNum1:-self.fNum2]
            else:
                ii_x = util.findNearest(self.x, x)
                ii_z = util.findNearest(self.zFull, z)
                return self.n[ii_x, ii_z]
    
    ### source functions ###
    def set_user_source_profile(self, method, z0=0, sVec=None, sFunc=None):
        """
        set the spatial source profile explicitly (no frequency / signal information)
        Precondition: index of refraction profile is already set
        
        Parameters
        ----------
        method : string
            'vector' for vector defined profile
            'func' for function defined profile
        z0 : float
            Precondition: z0>=0
            reference starting point for sVec (m). Initialized to 0 m
        sVec : array
            if method=='vector', defines the source profile as an array
            Precondition: spacing between elements is dz
            Postcondition: E(z=z0) = sVec[0], E(z=z0+dz) = sVec[1], ... , E(z>=z0+len(sVec)*dz) = sVec[-1], TODO
        sFunc : function
            if method=='func', defines the source profile as a function
            Precondition: sFunc is a function of one variable, z, and returns a float value
            Postcondition: E(z>=0) = sFunc(z)
        """    
        self.source = np.zeros(self.zNumFull, dtype='complex')
        
        ### vector method ###
        if method == 'vector': #TODO: Check that profile is consistent
            sNum = len(sVec)
            j = 0
            for i in range(self.zNumFull):
                if self.zFull[i] >= z0:
                    if j < sNum:
                        self.source[i] = sVec[j]
                    else:
                        self.source[i] = 0
                    j += 1
                else:
                    self.source[i] = 0
        
        ### functional method ###
        if method == 'func':
            for i in range(self.zNumFull):
                if self.zFull[i] >= 0:
                    self.source[i] = sFunc(self.zFull[i])
                else:
                    self.source[i] = 0      
        
    def set_dipole_source_profile(self, centerFreq, depth, A=1+0.j): #TODO Check if setting the centre frequency of model dipole is source of phase/time error!
        #TODO: Experiment with replacing centerFreq with a frequency array for TD simulations
        """
        set the source profile to be a half-wave dipole sized to center frequency
        Precondition: index of refraction profile is already set
        
        Parameters
        ----------  
        centerFreq : float
            center frequency of to model dipole around (GHz)
        depth : float
            Precondition: depth>=0
            depth of middle point of dipole (m)
        A : complex float
            complex amplitude of dipole. Initialized to 1 + 0j
        """
        ### frequency and wavelength in freespace ###
        self.source = np.zeros(self.zNumFull, dtype='complex')
        centerLmbda = util.c_light/centerFreq
        
        ### wavelength at reference depth ###
        centerLmbda0 = centerLmbda/abs(self.n0)
        
        ### create dipole ###
        z0 = depth
        z0Index = util.findNearest(self.zFull, z0)
        print(centerLmbda0, self.dz)
        nPoints = int((centerLmbda0/2) / self.dz)
        ZR1 = np.linspace(0,1, nPoints, dtype='complex')
        ZR2 = np.linspace(1,0, nPoints, dtype='complex')
        zRange = np.append(ZR1, ZR2)
        
        n_x = np.pi*zRange #TODO: What does this mean?
        e = [0., 0., 1.]
        beam = np.zeros(len(n_x), dtype='complex')
        f0 = np.zeros(len(n_x), dtype='complex')
        
        for i in range(len(n_x)):
            n=[n_x[i], 0, 0]
            val = np.cross(np.cross(n,e),n)[2]
            beam[i] = complex(val, val)
        f0 = A*(beam/(np.max(beam)))

        self.centerFreq = centerFreq
        self.source[z0Index-nPoints+1:z0Index+nPoints+1]=f0
    def set_phased_array(self, zstart, centerFreq, A=1+0j, n=8, scaling=1):
        '''

            Set a phased array string of antennas -> with some number of dipole antennas connected
            Old Parameters
            z: ? -> removed this is simply the depth array -> zFull or z??
            A: Amplitude
            n: Number of Dipoles on string
            scaling: ?


            From previous example: 'Deimnesions defined by wavelength
            My parameters:
            -zstart -> start depth
            -centerFreq -> central frequency

        '''
        #field0 = np.zeros(self.zNumFull, dtype='complex')

        wavel = util.c_light/centerFreq #Wavelength
        half_wavel = wavel/2 #Half of the Wavelength
        self.source = np.zeros(self.zNumFull, dtype='complex')
        for i in range(0, n):
            #print(i)
            z_i = zstart + (i*scaling*wavel) #position of antenna on string??
            ii_z_mid = util.findNearest(self.zFull, z_i)

            nPoints = int(half_wavel/self.dz) #points in HalfWavelength
            if nPoints > int(len(self.zFull)/4): # Legacy Code -> Don't know what this is
                nPoints = int(self.zNumFull/4)
            # zRange = np.linspace(-half_wavel, half_wavel, 2*nPoints, dtype='complex') #THIS IS DEFINED AGAIN -> Comment out
            zR1 = np.linspace(0,1, nPoints, dtype='complex')
            zR2 = np.linspace(1,0, nPoints, dtype='complex')
            zRange = np.append(zR1, zR2) #WHY IS ZRANGE DEFINED TWICE??
            n_x = np.pi * zRange
            e = [0,0,1]
            beam = np.zeros(len(n_x), dtype='complex')
            f0 = np.zeros(len(n_x), dtype='complex')
            for j in range(len(n_x)): # In paraProp phased array -> he uses the same integer twice!
                n_j = [n_x[j], 0, 0]
                val = (np.cross(np.cross(n_j,e), n_j)[2])
                beam[j] = complex(val, val)
            f0 = (A*(beam/np.max(beam)))
            self.source[ii_z_mid-nPoints+1:ii_z_mid+nPoints+1] = f0
    def phase_array_vector(self, zVec, sVec, centerFreq): #Define a phased array with a Vector of Antenna Positions and Amplitudes (includes phase!)
        nArray = len(sVec)
        self.source = np.zeros(self.zNumFull, dtype='complex')
        wavel = util.c_light/centerFreq #Wavelength
        half_wavel = wavel/2 #Half of the Wavelength
        for i in range(nArray):
            z_antenna = zVec[i] # The Depth of the antenna
            amp_antenna = sVec[i] #The Complex Amplitude of the Antenna
            ii_z_md = util.findNearest(self.zFull, z_antenna)
            nPoints = int(half_wavel/self.dz) #points in HalfWavelength
            if nPoints > int(len(self.zFull)/4): # Legacy Code -> Don't know what this is
                nPoints = int(self.zNumFull/4)
            # zRange = np.linspace(-half_wavel, half_wavel, 2*nPoints, dtype='complex') #THIS IS DEFINED AGAIN -> Comment out
            zR1 = np.linspace(0,1, nPoints, dtype='complex')
            zR2 = np.linspace(1,0, nPoints, dtype='complex')
            zRange = np.append(zR1, zR2) #WHY IS ZRANGE DEFINED TWICE??
            n_x = np.pi * zRange
            e = [0,0,1]
            beam = np.zeros(len(n_x), dtype='complex')
            f0 = np.zeros(len(n_x), dtype='complex')
            for j in range(len(n_x)): # In paraProp phased array -> he uses the same integer twice!
                n_j = [n_x[j], 0, 0]
                val = (np.cross(np.cross(n_j,e), n_j)[2])
                beam[j] = complex(val, val)
            f0 = amp_antenna * (beam/np.max(beam))
            self.source[ii_z_md-nPoints+1:ii_z_md+nPoints+1] = f0

    def get_source_profile(self):
        """
        gets source profile of simulation
        
        Returns
        -------
        1-d comlplex float array
        """
        return self.source[self.fNum1:-self.fNum2]
   
    
    ### signal functions ###
    def set_cw_source_signal(self, freq): #TODO: Consider changing the amplitude
        """
        set a continuous wave signal at a specified frequency
        
        Parameters
        ----------
        freq : float
            frequency of source (GHz) 
        """
        ### frequency ###
        self.freq = np.array([freq], dtype='complex')
        self.freqNum = len(self.freq)
        
        ### wavenumber at reference depth ###
        self.kp0 = 2.*np.pi*self.freq*self.n0/util.c_light
        self.k0 = 2*np.pi*self.freq/util.c_light
        
        ### coefficient ###
        self.A = np.array([1], dtype='complex')
        
    def set_td_source_signal(self, sigVec, dt):
        ### save input ###
        self.dt = dt
        self.sigVec = sigVec
        
        ### frequencies ###
        df = 1/(len(sigVec)*dt)
        self.freq = np.arange(0, 1/dt, df, dtype='complex')
        self.freqNum = len(self.freq)
        
        ### wavenumbers at reference depth ###
        self.kp0 = 2.*np.pi*self.freq*self.n0/util.c_light #Local Wavenumber k'0 = k0/n0
        self.k0 = 2*np.pi*self.freq/util.c_light #Vacuum Wavenumber

        ### coefficient ###
        self.A = util.doFFT(np.flip(sigVec)) #TODO: Check this
        
        # to ignore the DC component #
        self.A[0] = self.kp0[0] = 0

        
    def get_spectrum(self):
        """
        gets transmitted signal spectrum
        
        Returns
        -------
        1-d comlplex float array
        """
        return self.A[:int(self.freqNum/2)]
    
    def get_frequency(self):
        """
        gets frequency array
        
        Returns
        -------
        1-d float array
        """
        return abs(self.freq)[:int(self.freqNum/2)]
    
    def get_signal(self):
        """
        gets transmitted signal
        
        Returns
        -------
        1-d comlplex float array
        """
        return self.sigVec
    
    def get_time(self):
        """
        gets time array
        
        Returns
        -------
        1-d float array
        """
        return np.arange(0, self.dt*len(self.sigVec), self.dt)
               
        
    ### field functions ###


    def do_solver(self, rxList=np.array([]), freqMin=None, freqMax=None, solver_mode = 'one-way', refl_threshold=1e-10):
        """
        calculate field across the entire geometry (fd mode) or at receivers (td mode)
        field can be estimate in the forwards or backwards direction or in both directions
        -> modified from do_solver()
        -> calculates forwards field
        -> if dn/dx > 0 -> save position of reflector
        -> use as a source
        -> calculate an ensemble of u_minus

        Precondition: index of refraction and source profiles are set

        future implementation plans:
            - different method options
            - only store last range step option

        Parameters
        ----------
        Optional:
        -rxList : array of Receiver objects
            optional for cw signal simulation
            required for non cw signal (td) simulation
        -freqMin : float (must be less than nyquist frequnecy)
            defines minimum cutoff frequnecy for TD evalaution
        -freqMax : float (must be less than nyquist frequnecy)
            defines maximum cutoff frequuncy for TD evaluation
        -solver_mode : string
            defines the simulation mode
            must be one of three options:
                one-way : only evaluates in the forwards direction (+)
                two-way : evaluates in forwards (+) and backwards direction (-)
                minus : only evaluates in the backwards (-) direction
        -refl_threshold : float
            sets minimum reflection power to be simulated (anything less will be neglected)

        Output:
        FD mode: self.field has solution of E field across the simualtion geometry for the inputs : n, f and z_tx
        TD mode: rxList contains the signal and spectra for the array of receivers
        """
        if solver_mode != 'one-way' and solver_mode != 'two-way':
            print('warning! solver mode must be given as: one-way or two-way')
            print('will default to one way')
            solver_mode = 'one-way'
        if (self.freqNum != 1):
            ### check for Receivers ###
            if (len(rxList) == 0):
                print("Warning: Running time-domain simulation with no receivers. Field will not be saved.")
            for rx in rxList:
                rx.setup(self.freq, self.dt)
        print(solver_mode)
        #Check if solving for TD signal or in FD
        if freqMin == None and freqMax == None:
            freq_ints = np.arange(0, int(self.freqNum / 2) + self.freqNum % 2, 1, dtype='int')
        elif freqMin == None and freqMax != None:
            ii_min = util.findNearest(self.freq, freqMin)
            freq_ints = np.arange(ii_min, int(self.freqNum / 2) + self.freqNum % 2, 1, dtype='int')
        elif freqMin != None and freqMax == None:
            ii_max = util.findNearest(self.freq, freqMax)
            freq_ints = np.arange(0, ii_max, 1, dtype='int')
        else:
            ii_min = util.findNearest(self.freq, freqMin)
            ii_max = util.findNearest(self.freq, freqMax)
            freq_ints = np.arange(ii_min, ii_max, 1, dtype='int')

        for j in freq_ints:
            if (self.freq[j] == 0): continue
            u_plus = 2 * self.A[j] * self.source * self.filt * self.freq[j]
            self.field[0] = u_plus[self.fNum1:-self.fNum2]
            alpha_plus = np.exp(1.j * self.dx * self.kp0[j] * (np.sqrt(1. - (self.kz ** 2 / self.kp0[j] ** 2)) - 1.))
            B_plus = self.n ** 2 - 1
            Y_plus = np.sqrt(1. + (self.n / self.n0) ** 2)
            beta_plus = np.exp(1.j * self.dx * self.kp0[j] * (np.sqrt(B_plus + Y_plus ** 2) - Y_plus))

            if solver_mode == 'two-way':
                refl_source_list = []
                x_refl = []
                ix_refl = []
                nRefl = 0

            for i in range(1, self.xNum):
                u_plus_i = u_plus # Record starting reduced field -> in case we need to calculate
                dn = self.n[i,:] - self.n[i - 1,:]
                if dn.any() > 0:
                    u_plus *= util.transmission_coefficient(self.n[i], self.n[i-1])
                u_plus = alpha_plus * (util.doFFT(u_plus))
                u_plus = beta_plus[i] * (util.doIFFT(u_plus))
                u_plus = self.filt * u_plus
                delta_x_plus = self.dx * i
                self.field_plus[i] = u_plus[self.fNum1:-self.fNum2] / (
                        np.sqrt(delta_x_plus) * np.exp(-1.j * self.k0[j] * delta_x_plus))

                if solver_mode == 'two-way': #set to reflection modes
                    if dn.any() > 0: #check if the ref index changes in x direction
                        refl_source = util.reflection_coefficient(self.n[i], self.n[i-1]) * u_plus_i
                        if (refl_source**2).any() > refl_threshold: #check if reflected power is above threshold
                            x_refl.append(self.x[i])
                            refl_source_list.append(refl_source)
                            ix_refl.append(i)
                            nRefl = len(refl_source_list)
            if solver_mode == 'two-way' or solver_mode == 'minus':  # set to reflection modes
                if nRefl > 0:
                    u_minus_3arr = np.zeros((self.zNumFull, nRefl), dtype='complex')
                    field_minus_3arr = np.zeros((self.xNum, self.zNum, nRefl), dtype='complex')
                    for l in range(nRefl):
                        ix = ix_refl[l]
                        u_minus_3arr[:, l] = refl_source_list[l]
                        field_minus_3arr[ix, :, l] = u_minus_3arr[self.fNum1:-self.fNum2, l]
                        alpha_minus = np.exp(
                            1.j * self.dx * self.kp0[j] * (np.sqrt(1. - (self.kz ** 2 / self.kp0[j] ** 2)) - 1.))
                        B_minus = self.n ** 2 - 1
                        Y_minus = np.sqrt(1. + (self.n / self.n0) ** 2)
                        beta_minus = np.exp(1.j * self.dx * self.kp0[j] * (np.sqrt(B_minus + Y_minus ** 2) - Y_minus))
                        ix_last = ix_refl[l]
                        for k in np.flip(np.arange(0, ix_last, 1, dtype='int')):
                            x_minus = self.x[k]
                            dx_minus = abs(x_minus - self.x[ix_last])

                            u_minus_3arr[:, l] = alpha_minus * (util.doFFT(u_minus_3arr[:, l]))  # ????
                            u_minus_3arr[:, l] = beta_minus[k] * (util.doIFFT(u_minus_3arr[:, l]))
                            u_minus_3arr[:, l] = self.filt * u_minus_3arr[:, l]
                            field_minus_3arr[k, :, l] = np.transpose(
                                (u_minus_3arr[self.fNum1:-self.fNum2, l] / np.sqrt(dx_minus)) * np.exp(
                                    1j * dx_minus * self.k0[j]))
                        self.field_minus[:,:] += field_minus_3arr[:,:,l]

            if solver_mode == 'one-way':
                self.field[:,:] = self.field_plus[:,:]
                if (len(rxList) != 0):
                    for rx in rxList:
                        rx.add_spectrum_component(self.freq[j], self.get_field(x0=rx.x, z0=rx.z))
                    self.field.fill(0)
            elif solver_mode == 'two-way':
                if (len(rxList) != 0):
                    for rx in rxList:
                        rx.add_spectrum_component_plus(self.freq[j], self.get_field_plus(x0=rx.x, z0=rx.z))
                        rx.add_spectrum_component_minus(self.freq[j], self.get_field_minus(x0=rx.x, z0=rx.z))
                        rx.spectrum = rx.spectrum_plus + rx.spectrum_minus
                    self.field.fill(0)
                    self.field_plus.fill(0)
                    self.field_minus.fill(0)
                else:
                    self.field[:,:] += self.field_plus[:,:]
                    self.field[:,:] += self.field_minus[:,:]
    def do_solver_smooth(self, rxList=np.array([]), ant_length=1.0, freqMin=None, freqMax=None, solver_mode = 'one-way', refl_threshold=1e-10):
        """
        calculate field across the entire geometry (fd mode) or at receivers (td mode)
        field can be estimate in the forwards or backwards direction or in both directions
        -> modified from do_solver()
        -> calculates forwards field
        -> if dn/dx > 0 -> save position of reflector
        -> use as a source
        -> calculate an ensemble of u_minus

        Precondition: index of refraction and source profiles are set

        Change:
        Smooth rx over a list of locations

        future implementation plans:
            - different method options
            - only store last range step option

        Parameters
        ----------
        Optional:
        -rxList : array of Receiver objects
            optional for cw signal simulation
            required for non cw signal (td) simulation
        -freqMin : float (must be less than nyquist frequnecy)
            defines minimum cutoff frequnecy for TD evalaution
        -freqMax : float (must be less than nyquist frequnecy)
            defines maximum cutoff frequuncy for TD evaluation
        -solver_mode : string
            defines the simulation mode
            must be one of three options:
                one-way : only evaluates in the forwards direction (+)
                two-way : evaluates in forwards (+) and backwards direction (-)
                minus : only evaluates in the backwards (-) direction
        -refl_threshold : float
            sets minimum reflection power to be simulated (anything less will be neglected)

        Output:
        FD mode: self.field has solution of E field across the simualtion geometry for the inputs : n, f and z_tx
        TD mode: rxList contains the signal and spectra for the array of receivers
        """
        if solver_mode != 'one-way' and solver_mode != 'two-way':
            print('warning! solver mode must be given as: one-way or two-way')
            print('will default to one way')
            solver_mode = 'one-way'
        if (self.freqNum != 1):
            ### check for Receivers ###
            if (len(rxList) == 0):
                print("Warning: Running time-domain simulation with no receivers. Field will not be saved.")
            for rx in rxList:
                rx.setup(self.freq, self.dt)
        print(solver_mode)
        #Check if solving for TD signal or in FD
        if freqMin == None and freqMax == None:
            freq_ints = np.arange(0, int(self.freqNum / 2) + self.freqNum % 2, 1, dtype='int')
        elif freqMin == None and freqMax != None:
            ii_min = util.findNearest(self.freq, freqMin)
            freq_ints = np.arange(ii_min, int(self.freqNum / 2) + self.freqNum % 2, 1, dtype='int')
        elif freqMin != None and freqMax == None:
            ii_max = util.findNearest(self.freq, freqMax)
            freq_ints = np.arange(0, ii_max, 1, dtype='int')
        else:
            ii_min = util.findNearest(self.freq, freqMin)
            ii_max = util.findNearest(self.freq, freqMax)
            freq_ints = np.arange(ii_min, ii_max, 1, dtype='int')

        for j in freq_ints:
            #print(abs(j/len(self.freq))*200,  '%', abs(self.freq[j]*1e3))

            if (self.freq[j] == 0): continue
            u_plus = 2 * self.A[j] * self.source * self.filt * self.freq[j]
            self.field[0] = u_plus[self.fNum1:-self.fNum2]

            alpha_plus = np.exp(1.j * self.dx * self.kp0[j] * (np.sqrt(1. - (self.kz ** 2 / self.kp0[j] ** 2)) - 1.))
            B_plus = self.n ** 2 - 1
            Y_plus = np.sqrt(1. + (self.n / self.n0) ** 2)
            beta_plus = np.exp(1.j * self.dx * self.kp0[j] * (np.sqrt(B_plus + Y_plus ** 2) - Y_plus))

            if solver_mode == 'two-way':
                refl_source_list = []
                x_refl = []
                ix_refl = []
                nRefl = 0

            for i in range(1, self.xNum):
                u_plus_i = u_plus # Record starting reduced field -> in case we need to calculate
                dn = self.n[i] - self.n[i - 1]
                if dn.any() > 0:
                    u_plus *= util.transmission_coefficient(self.n[i], self.n[i-1])
                u_plus = alpha_plus * (util.doFFT(u_plus))
                u_plus = beta_plus[i] * (util.doIFFT(u_plus))
                u_plus = self.filt * u_plus
                delta_x_plus = self.dx * i
                self.field_plus[i] = u_plus[self.fNum1:-self.fNum2] / (
                        np.sqrt(delta_x_plus) * np.exp(-1.j * self.k0[j] * delta_x_plus))

                if solver_mode == 'two-way': #set to reflection modes
                    if dn.any() > 0: #check if the ref index changes in x direction
                        refl_source = util.reflection_coefficient(self.n[i], self.n[i-1]) * u_plus_i
                        if (refl_source**2).any() > refl_threshold: #check if reflected power is above threshold
                            x_refl.append(self.x[i])
                            refl_source_list.append(refl_source)
                            ix_refl.append(i)
                            nRefl = len(refl_source_list)
            if solver_mode == 'two-way' or solver_mode == 'minus':  # set to reflection modes
                if nRefl > 0:
                    u_minus_3arr = np.zeros((self.zNumFull, nRefl), dtype='complex')
                    field_minus_3arr = np.zeros((self.xNum, self.zNum, nRefl), dtype='complex')
                    for l in range(nRefl):
                        ix = ix_refl[l]
                        u_minus_3arr[:, l] = refl_source_list[l]
                        field_minus_3arr[ix, :, l] = u_minus_3arr[self.fNum1:-self.fNum2, l]
                        alpha_minus = np.exp(
                            1.j * self.dx * self.kp0[j] * (np.sqrt(1. - (self.kz ** 2 / self.kp0[j] ** 2)) - 1.))
                        B_minus = self.n ** 2 - 1
                        Y_minus = np.sqrt(1. + (self.n / self.n0) ** 2)
                        beta_minus = np.exp(1.j * self.dx * self.kp0[j] * (np.sqrt(B_minus + Y_minus ** 2) - Y_minus))
                        ix_last = ix_refl[l]
                        for k in np.flip(np.arange(0, ix_last, 1, dtype='int')):
                            x_minus = self.x[k]
                            dx_minus = abs(x_minus - self.x[ix_last])

                            u_minus_3arr[:, l] = alpha_minus * (util.doFFT(u_minus_3arr[:, l]))  # ????
                            u_minus_3arr[:, l] = beta_minus[k] * (util.doIFFT(u_minus_3arr[:, l]))
                            u_minus_3arr[:, l] = self.filt * u_minus_3arr[:, l]
                            field_minus_3arr[k, :, l] = np.transpose(
                                (u_minus_3arr[self.fNum1:-self.fNum2, l] / np.sqrt(dx_minus)) * np.exp(
                                    1j * dx_minus * self.k0[j]))
                        self.field_minus[:,:] += field_minus_3arr[:,:,l]

            if solver_mode == 'one-way':
                self.field[:,:] = self.field_plus[:,:]
                if (len(rxList) != 0):
                    for rx in rxList:
                        z_centre = rx.z
                        z_0 = z_centre - ant_length/2
                        z_1 = z_centre + ant_length/2
                        z_sample_range = np.arange(z_0, z_1 + self.dz, self.dz)
                        nZ = len(z_sample_range)
                        amp_at_antenna = 0
                        for n in range(nZ):
                            amp_at_antenna += self.get_field(x0=rx.x,z0=z_sample_range[n])
                        amp_at_antenna /= float(nZ)
                        rx.add_spectrum_component(self.freq[j], amp_at_antenna)
                    self.field.fill(0)
            elif solver_mode == 'two-way':
                if (len(rxList) != 0):
                    for rx in rxList:
                        rx.add_spectrum_component_plus(self.freq[j], self.get_field_plus(x0=rx.x, z0=rx.z))
                        rx.add_spectrum_component_minus(self.freq[j], self.get_field_minus(x0=rx.x, z0=rx.z))
                        rx.spectrum = rx.spectrum_plus + rx.spectrum_minus
                    self.field.fill(0)
                    self.field_plus.fill(0)
                    self.field_minus.fill(0)
                else:
                    self.field[:,:] += self.field_plus[:,:]
                    self.field[:,:] += self.field_minus[:,:]

    def do_solver_smooth2(self, rxList=np.array([]), ant_length=1.0, freqMin=None, freqMax=None, solver_mode = 'one-way', refl_threshold=1e-10):
        """
        calculate field across the entire geometry (fd mode) or at receivers (td mode)
        field can be estimate in the forwards or backwards direction or in both directions
        -> modified from do_solver()
        -> calculates forwards field
        -> if dn/dx > 0 -> save position of reflector
        -> use as a source
        -> calculate an ensemble of u_minus

        Precondition: index of refraction and source profiles are set

        Change:
        Smooth rx over a list of locations

        future implementation plans:
            - different method options
            - only store last range step option

        Parameters
        ----------
        Optional:
        -rxList : array of Receiver objects
            optional for cw signal simulation
            required for non cw signal (td) simulation
        -freqMin : float (must be less than nyquist frequnecy)
            defines minimum cutoff frequnecy for TD evalaution
        -freqMax : float (must be less than nyquist frequnecy)
            defines maximum cutoff frequuncy for TD evaluation
        -solver_mode : string
            defines the simulation mode
            must be one of three options:
                one-way : only evaluates in the forwards direction (+)
                two-way : evaluates in forwards (+) and backwards direction (-)
                minus : only evaluates in the backwards (-) direction
        -refl_threshold : float
            sets minimum reflection power to be simulated (anything less will be neglected)

        Output:
        FD mode: self.field has solution of E field across the simualtion geometry for the inputs : n, f and z_tx
        TD mode: rxList contains the signal and spectra for the array of receivers
        """
        if solver_mode != 'one-way' and solver_mode != 'two-way':
            print('warning! solver mode must be given as: one-way or two-way')
            print('will default to one way')
            solver_mode = 'one-way'
        if (self.freqNum != 1):
            ### check for Receivers ###
            if (len(rxList) == 0):
                print("Warning: Running time-domain simulation with no receivers. Field will not be saved.")
            for rx in rxList:
                rx.setup(self.freq, self.dt)
        print(solver_mode)
        #Check if solving for TD signal or in FD
        if freqMin == None and freqMax == None:
            freq_ints = np.arange(0, int(self.freqNum / 2) + self.freqNum % 2, 1, dtype='int')
        elif freqMin == None and freqMax != None:
            ii_min = util.findNearest(self.freq, freqMin)
            freq_ints = np.arange(ii_min, int(self.freqNum / 2) + self.freqNum % 2, 1, dtype='int')
        elif freqMin != None and freqMax == None:
            ii_max = util.findNearest(self.freq, freqMax)
            freq_ints = np.arange(0, ii_max, 1, dtype='int')
        else:
            ii_min = util.findNearest(self.freq, freqMin)
            ii_max = util.findNearest(self.freq, freqMax)
            freq_ints = np.arange(ii_min, ii_max, 1, dtype='int')

        for j in freq_ints:
            #print(abs(j/len(self.freq))*200,  '%', abs(self.freq[j]*1e3))

            if (self.freq[j] == 0): continue
            u_plus = 2 * self.A[j] * self.source * self.filt * self.freq[j]
            self.field[0] = u_plus[self.fNum1:-self.fNum2]

            alpha_plus = np.exp(1.j * self.dx * self.kp0[j] * (np.sqrt(1. - (self.kz ** 2 / self.kp0[j] ** 2)) - 1.))
            B_plus = self.n ** 2 - 1
            Y_plus = np.sqrt(1. + (self.n / self.n0) ** 2)
            beta_plus = np.exp(1.j * self.dx * self.kp0[j] * (np.sqrt(B_plus + Y_plus ** 2) - Y_plus))

            if solver_mode == 'two-way':
                refl_source_list = []
                x_refl = []
                ix_refl = []
                nRefl = 0

            for i in range(1, self.xNum):
                u_plus_i = u_plus # Record starting reduced field -> in case we need to calculate
                dn = self.n[i] - self.n[i - 1]
                if dn.any() > 0:
                    u_plus *= util.transmission_coefficient(self.n[i], self.n[i-1])
                u_plus = alpha_plus * (util.doFFT(u_plus))
                u_plus = beta_plus[i] * (util.doIFFT(u_plus))
                u_plus = self.filt * u_plus

                delta_x_plus = self.dx * i
                self.field_plus[i] = u_plus[self.fNum1:-self.fNum2] / (
                        np.sqrt(delta_x_plus) * np.exp(-1.j * self.k0[j] * delta_x_plus))

                if solver_mode == 'two-way': #set to reflection modes
                    if dn.any() > 0: #check if the ref index changes in x direction
                        refl_source = util.reflection_coefficient(self.n[i], self.n[i-1]) * u_plus_i
                        if (refl_source**2).any() > refl_threshold: #check if reflected power is above threshold
                            x_refl.append(self.x[i])
                            refl_source_list.append(refl_source)
                            ix_refl.append(i)
                            nRefl = len(refl_source_list)
            if solver_mode == 'two-way' or solver_mode == 'minus':  # set to reflection modes
                if nRefl > 0:
                    u_minus_3arr = np.zeros((self.zNumFull, nRefl), dtype='complex')
                    field_minus_3arr = np.zeros((self.xNum, self.zNum, nRefl), dtype='complex')
                    for l in range(nRefl):
                        ix = ix_refl[l]
                        u_minus_3arr[:, l] = refl_source_list[l]
                        field_minus_3arr[ix, :, l] = u_minus_3arr[self.fNum1:-self.fNum2, l]
                        alpha_minus = np.exp(
                            1.j * self.dx * self.kp0[j] * (np.sqrt(1. - (self.kz ** 2 / self.kp0[j] ** 2)) - 1.))
                        B_minus = self.n ** 2 - 1
                        Y_minus = np.sqrt(1. + (self.n / self.n0) ** 2)
                        beta_minus = np.exp(1.j * self.dx * self.kp0[j] * (np.sqrt(B_minus + Y_minus ** 2) - Y_minus))
                        ix_last = ix_refl[l]
                        for k in np.flip(np.arange(0, ix_last, 1, dtype='int')):
                            x_minus = self.x[k]
                            dx_minus = abs(x_minus - self.x[ix_last])

                            u_minus_3arr[:, l] = alpha_minus * (util.doFFT(u_minus_3arr[:, l]))  # ????
                            u_minus_3arr[:, l] = beta_minus[k] * (util.doIFFT(u_minus_3arr[:, l]))
                            u_minus_3arr[:, l] = self.filt * u_minus_3arr[:, l]
                            field_minus_3arr[k, :, l] = np.transpose(
                                (u_minus_3arr[self.fNum1:-self.fNum2, l] / np.sqrt(dx_minus)) * np.exp(
                                    1j * dx_minus * self.k0[j]))
                        self.field_minus[:,:] += field_minus_3arr[:,:,l]

            if solver_mode == 'one-way':
                self.field[:,:] = self.field_plus[:,:]
                if (len(rxList) != 0):
                    for rx in rxList:
                        z_centre = rx.z
                        z_0 = z_centre - ant_length/2
                        z_1 = z_centre + ant_length/2
                        z_sample_range = np.arange(z_0, z_1 + self.dz, self.dz)
                        nZ = len(z_sample_range)
                        amp_at_antenna = 0
                        for n in range(nZ):
                            amp_n = self.get_field(x0=rx.x,z0=z_sample_range[n])
                            delta_z = z_sample_range[n] - z_centre
                            eff = np.cos(pi*(delta_z/ant_length))
                            #amp_n *= (eff/2*self.dz)
                            amp_n *= (eff/(4*pi))

                            amp_at_antenna += amp_n
                        rx.add_spectrum_component(self.freq[j], amp_at_antenna)
                    self.field.fill(0)
            elif solver_mode == 'two-way':
                if (len(rxList) != 0):
                    for rx in rxList:
                        rx.add_spectrum_component_plus(self.freq[j], self.get_field_plus(x0=rx.x, z0=rx.z))
                        rx.add_spectrum_component_minus(self.freq[j], self.get_field_minus(x0=rx.x, z0=rx.z))
                        rx.spectrum = rx.spectrum_plus + rx.spectrum_minus
                    self.field.fill(0)
                    self.field_plus.fill(0)
                    self.field_minus.fill(0)
                else:
                    self.field[:,:] += self.field_plus[:,:]
                    self.field[:,:] += self.field_minus[:,:]


    def get_field_minus(self, x0=None, z0=None):
        """
                gets field calculated by simulation

                future implementation plans:
                    - interpolation option
                    - specify complex, absolute, real, or imaginary field

                Parameters
                ----------
                x0 : float
                    position of interest in x-dimension (m). optional
                z0 : float
                    position of interest in z-dimension (m). optional

                Returns
                -------
                if both x0 and z0 are supplied
                    complex float
                if only one of x0 or z0 is supplied
                    1-d complex float array
                if neither x0 or z0 are supplied
                    2-d complex float array
                """
        if (x0 != None and z0 != None):
            return self.field_minus[util.findNearest(self.x, x0), util.findNearest(self.z, z0)]
        if (x0 != None and z0 == None):
            return self.field_minus[util.findNearest(self.x, x0), :]
        if (x0 == None and z0 != None):
            return self.field_minus[:, util.findNearest(self.z, z0)]
        return self.field_minus

    def get_field(self, x0=None, z0=None):
        """
        gets field calculated by simulation

        future implementation plans:
            - interpolation option
            - specify complex, absolute, real, or imaginary field

        Parameters
        ----------
        x0 : float
            position of interest in x-dimension (m). optional
        z0 : float
            position of interest in z-dimension (m). optional

        Returns
        -------
        if both x0 and z0 are supplied
            complex float
        if only one of x0 or z0 is supplied
            1-d complex float array
        if neither x0 or z0 are supplied
            2-d complex float array
        """
        if (x0 != None and z0 != None):
            return self.field[util.findNearest(self.x, x0), util.findNearest(self.z, z0)]
        if (x0 != None and z0 == None):
            return self.field[util.findNearest(self.x, x0), :]
        if (x0 == None and z0 != None):
            return self.field[:, util.findNearest(self.z, z0)]
        return self.field

    def get_field_plus(self, x0=None, z0=None):
        """
                        gets field calculated by simulation

                        future implementation plans:
                            - interpolation option
                            - specify complex, absolute, real, or imaginary field

                        Parameters
                        ----------
                        x0 : float
                            position of interest in x-dimension (m). optional
                        z0 : float
                            position of interest in z-dimension (m). optional

                        Returns
                        -------
                        if both x0 and z0 are supplied
                            complex float
                        if only one of x0 or z0 is supplied
                            1-d complex float array
                        if neither x0 or z0 are supplied
                            2-d complex float array
                        """
        if (x0 != None and z0 != None):
            return self.field_plus[util.findNearest(self.x, x0), util.findNearest(self.z, z0)]
        if (x0 != None and z0 == None):
            return self.field_plus[util.findNearest(self.x, x0), :]
        if (x0 == None and z0 != None):
            return self.field_plus[:, util.findNearest(self.z, z0)]
        return self.field_plus

    ### misc. functions ###
    def at_depth(self, vec, depth):
        """
        find value of vector at specified depth.
        future implementation plans:
            - interpolation option
            - 2D array seraching. paraProp.at_depth() -> paraProp.at()
        
        Parameters
        ----------
        vec : array
            vector of values
            Precondition: len(vec) = len(z)
        depth : float
            depth of interest (m)
        
        Returns
        -------
        base type of vec
        """  
        ### error if depth is out of bounds of simulation ###
        if (depth > self.iceDepth or depth < -self.airHeight):
                print("Error: Looking at z-position of out bounds")
                return np.NaN
         
        # find closest index #
        dIndex = round((depth + self.fNum1*self.dz + self.airHeight) / self.dz)
        
        return vec[int(dIndex)]