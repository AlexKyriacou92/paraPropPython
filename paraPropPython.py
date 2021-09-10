# paraPropPython
# c. sbrocco, s. prohira

import util
import numpy as np
import time
import datetime
from permittivity import *

class receiver:
    """
    Parameters
    ----------
    x : float
        x position (m)
    z : float
        z position (m)
    """
    def __init__(self, x, z):
        self.x = x
        self.z = z
    
    def setup(self, freq, dt):
        """
        further setup of receiver using simulation parameters
        
        Parameters
        ----------
        freq : float array
            frequencies (GHz)
        dt : float
            time step (ns)
        """
        self.freq = freq
        self.spectrum = np.zeros(len(freq), dtype='complex')
        self.time = np.arange(0, dt*len(freq), dt)
    
    def add_spectrum_component(self, f, A):
        """
        adds the contribution of a frequency to the received signal spectrum
        
        Parameters
        ----------
        f : float
            corresponding frequencie (GHz)
        A : complex float
            complex amplitude of received siganl (V/m???)
        """
        i = util.findNearest(self.freq, f)
        self.spectrum[i] = A
        
    def get_spectrum(self):
        """
        gets received signal spectrum
        
        Returns
        -------
        1-d comlplex float array
        """
        return self.spectrum[:int(len(self.freq)/2)]
    
    def get_signal(self):
        """
        gets received signal
        
        Returns
        -------
        1-d comlplex float array
        """
        return np.flip(util.doIFFT(self.spectrum))
    
    def get_frequency(self):
        """
        gets frequency array
        
        Returns
        -------
        1-d float array
        """
        return abs(self.freq)[:int(len(self.freq)/2)]
    
    def get_time(self):
        """
        gets time array
        
        Returns
        -------
        1-d float array
        """
        return self.time
     
        

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
        self.iceLength = iceLength
        self.refDepth = refDepth
        self.iceDepth = iceDepth
        self.airHeight = airHeight
        self.z = np.arange(-airHeight, iceDepth + dz, dz)
        self.zFull = np.arange(-(airHeight + filterDepth), iceDepth + filterDepth + dz, dz)
        self.zNum = len(self.z)
        self.zNumFull = len(self.zFull)
        self.dz = dz
        self.refDepth = refDepth            
        
        ### other simulation variables ###       
        # filter information #
        self.filterDepth = filterDepth
        self.fNum = int(filterDepth / dz)
        win = np.blackman(2*self.fNum)
        filt = np.ones(self.zNumFull)
        filt[:self.fNum] = win[:self.fNum]
        filt[-self.fNum:] = win[self.fNum:]
        self.filt = filt
       
        # z wavenumber #
        self.kz = np.zeros(self.zNumFull)
        self.kz[:int(self.zNumFull/2)] = np.linspace(0, np.pi/self.dz, int(self.zNumFull/2))
        self.kz[-int(self.zNumFull/2):] = np.linspace(-np.pi/self.dz, 0, int(self.zNumFull/2))
        
        # index of refraction array #
        self.n = np.ones(self.zNumFull, dtype='complex')

        #2D Refractive Index Profile - Added by Alex Kyriacou
        self.n2 = (1 + 1j) * np.zeros((self.xNum, self.zNumFull))
        
        # source array #
        self.source = np.zeros(self.zNumFull, dtype='complex')
        
        # 2d field array #
        self.field = np.zeros((self.xNum, self.zNum), dtype='complex')
        
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
    def set_n(self, method, nVec=None, nFunc=None, nAir=1.0003):
        """
        set the index of refraction profile of the simualtion
        
        future implementation plans:
            - 2-d profiles
            - complex index of refraction
        
        Parameters
        ----------
        method : string
            'vector' for vector defined profile
            'func' for function defined profile
        nVec : array
            if method=='vector', defines the index of refraction profile of ice as an array
            Precondition: spacing between elements is dz
            Postcondition: n(z=0) = nVec[0], n(z=dz) = nVec[1], ... , n(z>=len(nVec)*dz) = nVec[-1]
        nFunc : function
            if method=='func', defines the index of refraction profile of ice as a function
            Precondition: nFunc is a function of one variable, z, and returns a float value
            Postcondition: n(z>=0) = nFunc(z)
        nAir : float
            index of refraction of air
            Postcondition: n(z<0) = nAir
        """    
        self.n = np.ones(self.zNumFull, dtype='complex')
        
        ### vector method ###
        if method == 'vector': 
            nNum = len(nVec)
            j = 0
            for i in range(self.zNumFull):
                if self.zFull[i] >= 0:
                    if j < nNum:
                        self.n[i] = nVec[j]
                    else:
                        self.n[i] = nVec[-1]
                    j += 1
                else:
                    self.n[i] = nAir
         
        ### functional method ###
        if method == 'func':
            for i in range(self.zNumFull):
                if self.zFull[i] >= 0:
                    if self.zFull[i] <= self.iceDepth:
                        self.n[i] = nFunc(self.zFull[i])
                    else:
                        self.n[i] = nFunc(self.iceDepth)
                else:
                    self.n[i] = nAir
                    
        ### set reference index of refraction ###
        self.n0 = self.at_depth(self.n, self.refDepth)
        
    def get_n(self):
        """
        gets index of refraction profile of simulation
        
        Returns
        -------
        1-d float array
        """
        return self.n[self.fNum:-self.fNum]

   #Add 2D Refractive Index Profile -> Alex Kyriacou
    def set_n2(self, method, nMat=None, nFunc=None, nAir=1.0003):
        """
                set the index of refraction profile of the simualtion

                        future implementation plans:
                            - 2-d profiles
                            - complex index of refraction

                        Parameters
                        ----------
                        method : string
                            'vector' for vector defined profile
                            'func' for function defined profile
                        nVec : array
                            if method=='vector', defines the index of refraction profile of ice as an array
                            Precondition: spacing between elements is dz
                            Postcondition: n(z=0) = nVec[0], n(z=dz) = nVec[1], ... , n(z>=len(nVec)*dz) = nVec[-1]
                        nMat : Array
                            if method == 'matrix' defines a matrix containing the complex index of refracting profile of ice
                        nFunc : function
                            if method=='func', defines the index of refraction profile of ice as a function
                            Precondition: nFunc is a function of one variable, z, and returns a float value
                            Postcondition: n(z>=0) = nFunc(z)
                        nAir : float
                            index of refraction of air
                            Postcondition: n(z<0) = nAir
                        """
        # self.n2 = np.ones((self.xNum, self.zNumFull), dtype='complex')
        self.n2 = np.zeros((self.xNum, self.zNumFull), dtype ='complex')
        ### vector method ###

        if method == 'matrix':
            #print(self.n2.shape, nMat.shape)

            for i in range(self.xNum):
                self.n2[i, :] = nMat[i, :]  # Note that the vector will have to include the air surface as well

        ### functional method ###
        if method == 'func':
            for i in range(self.xNum):
                for j in range(self.zNumFull):
                    self.n2[i, j] = nFunc(self.x[i], self.zFull[j])

        ### set reference index of refraction ###
        self.n0 = self.at_depth(self.n, self.refDepth)
        # self.n0 = self.at(self.n2, self.refDepth, self.refRange)

    def get_n2(self):
        """
        gets index of refraction profile of simulation

        Returns
        -------
        1-d float array
        """
        return self.n2

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
        if method == 'vector':
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
        
    def set_dipole_source_profile(self, centerFreq, depth, A=1+0.j):
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
        centerLmbda0 = centerLmbda/self.n0
        
        ### create dipole ###
        z0 = depth
        z0Index = util.findNearest(self.zFull, z0)
        
        nPoints = int((centerLmbda0/2) / self.dz)
        ZR1 = np.linspace(0,1, nPoints, dtype='complex')
        ZR2 = np.linspace(1,0, nPoints, dtype='complex')
        zRange = np.append(ZR1, ZR2)
        
        n_x = np.pi*zRange
        e = [0., 0., 1.]
        beam = np.zeros(len(n_x), dtype='complex')
        f0 = np.zeros(len(n_x), dtype='complex')
        
        for i in range(len(n_x)):
            n=[n_x[i], 0, 0]
            val = np.cross(np.cross(n,e),n)[2]
            beam[i] = complex(val, val)
        f0 = A*(beam/(np.max(beam)))
        
        self.source[z0Index-nPoints+1:z0Index+nPoints+1]=f0
        
    def get_source_profile(self):
        """
        gets source profile of simulation
        
        Returns
        -------
        1-d comlplex float array
        """
        return self.source[self.fNum:-self.fNum]
   
    
    ### signal functions ###
    def set_cw_source_signal(self, freq, amplitude = 1+0j):
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
        self.k0 = 2.*np.pi*self.freq*self.n0/util.c_light 
        
        ### coefficient ###
        self.A = np.array([amplitude], dtype='complex')
        
    def set_td_source_signal(self, sigVec, dt):
        ### save input ###
        self.dt = dt
        self.sigVec = sigVec
        
        ### frequencies ###
        df = 1/(len(sigVec)*dt)
        self.freq = np.arange(0, 1/dt, df, dtype='complex') #TODO -> Why do they structure it like this?? why isn't the frequnecy space from -nyquist to +nyquist??
        self.freqNum = len(self.freq)
        
        ### wavenumbers at reference depth ###
        self.k0 = 2.*np.pi*self.freq*self.n0/util.c_light 
        
        ### coefficient ###
        self.A = util.doFFT(np.flip(sigVec))
        
        # to ignore the DC component #
        self.A[0] = self.k0[0] = 0

        
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
    def do_solver(self, rxList=np.array([])):
        """
        calculates field at points in the simulation
        Precondition: index of refraction and source profiles are set

        future implementation plans:
            - different method options
            - only store last range step option
            
        Parameters
        ----------
        rxList : array of Receiver objects
            optional for cw signal simulation
            required for non cw signal simulation
        """ 
        
        if (self.freqNum != 1):
            ### check for Receivers ###
            if (len(rxList) == 0):
                #print("Warning: Running time-domain simulation with no receivers. Field will not be saved.")
                pass
            for rx in rxList:
                rx.setup(self.freq, self.dt)
                
        for j in np.arange(0, int(self.freqNum/2)+self.freqNum%2, 1, dtype='int'):
            if (self.freq[j] == 0): continue
            u = 2 * self.A[j] * self.source * self.filt * self.freq[j] #Set reduced field u(0, z, f = f_j)
            self.field[0,:] = u[self.fNum:-self.fNum] #Set Field at x=0 psi(x=0, z, f = f_j)

            ### method II ###
            alpha = np.exp(1.j * self.dx * self.k0[j] * (np.sqrt(1. - (self.kz**2 / self.k0[j]**2))- 1.))
            B = (self.n)**2-1 #TODO Check if n is real or complex number -> does this change the maths??
            Y = np.sqrt(1.+(self.n/self.n0)**2)
            beta = np.exp(1.j * self.dx * self.k0[j] * (np.sqrt(B+Y**2)-Y))

            for i in range(1, self.xNum):           
                u = alpha * (util.doFFT(u))
                u = beta * (util.doIFFT(u))
                u = self.filt * u

                self.field[i,:] = u[self.fNum:-self.fNum]/(np.sqrt(self.dx*i) * np.exp(-1.j * self.k0[j] * self.dx * i))
            if (len(rxList) != 0):
                for rx in rxList:
                    rx.add_spectrum_component(self.freq[j], self.get_field(x0=rx.x, z0=rx.z))
                self.field.fill(0)

    def do_solver2(self, rxList=np.array([]), freq_min = 0, freq_max = 1, nDiv = 1):
        """
            calculates field at points in the simulation
            Precondition: index of refraction and source profiles are set

            future implementation plans:

                - different method options
                - only store last range step option

            Parameters
            ----------
            rxList : array of Receiver objects
                optional for cw signal simulation
                required for non cw signal simulation
        """

        if self.freqNum > 1:
            idx_min = util.findNearest(freq_min, self.freq)
            idx_max = util.findNearest(freq_max, self.freq)
        else:
            idx_min = 0
            idx_max = 1
        if (self.freqNum != 1):
            ### check for Receivers ###
            if (len(rxList) == 0):
                print("Warning: Running time-domain simulation with no receivers. Field will not be saved.")
            for rx in rxList:
                rx.setup(self.freq, self.dt)

        #TODO: Add Sinc Interpolation:

        #for k in range(1,self.freqNum):
        for k in range(idx_min, idx_max):
            if k % nDiv == 0:
                tstart_k = time.time()
                u = self.A[k] * self.source * self.filt
                #print('u shape', u.shape)

                self.field[0, :] = u[self.fNum:-self.fNum]
                # print( u[self.fNum:-self.fNum].shape)
                ### method II ###
                # print(self.kz)

                '''
                alpha = np.exp(1.j * self.dx * self.k0[k] * (np.sqrt(1. - (self.kz ** 2 / self.k0[k] ** 2)) - 1.))
                B = (self.n2) ** 2 - 1
                Y = np.sqrt(1. + (self.n2 / self.n0) ** 2)
    
                beta = np.exp(1.j * self.dx * self.k0[k] * (np.sqrt(B + Y ** 2) - Y))
                '''
                range_times = []
                for i in range(1, self.xNum):
                    tstart_range = time.time()
                    nVec = self.n2[i, :]
                    # Added by Alex Kyriacou
                    alpha = np.exp(1.j * self.dx * self.k0[k] * (
                                np.sqrt(1. - (self.kz ** 2 / self.k0[k] ** 2)) - 1.))  # This is a 1D vector
                    B = nVec ** 2 - 1
                    Y = np.sqrt(1. + (nVec / self.n0) ** 2)
                    beta = np.exp(1.j * self.dx * self.k0[k] * (np.sqrt(B + Y ** 2) - Y))

                    # TODO: Finish correcting this to be 2D -> fix tab
                    # u = alpha[i, :] * (util.doFFT(u))
                    u = alpha * (util.doFFT(u))
                    u = beta * (util.doIFFT(u))
                    u = self.filt * u

                    self.field[i, :] = u[self.fNum:-self.fNum] / (
                                np.sqrt(self.dx * i) * np.exp(-1.j * self.k0[k] * self.dx * i))
                    # print(self.x[i], self.at_depth(self.field[i, :], 40))
                    tend_range = time.time()
                    range_times.append(tend_range-tstart_range)
                range_time = np.mean(range_times)
                if (len(rxList) != 0):
                    for rx in rxList:
                        rx.add_spectrum_component(self.freq[k], self.get_field(x0=rx.x, z0=rx.z))
                    # self.field.fill(0) #Deletes Field Afterwards
                tend_k = time.time()
                duration = tend_k - tstart_k
                #print('Solution complete, time: ', round(duration, 2), 's')
                #print('Average time per range step: ', range_time, ' approximate solution for freq step: ', range_time*self.xNum)
                nRemaining = (idx_max - k) / nDiv
                #print('Remaining Iterations', nRemaining)
                remainder = datetime.timedelta(seconds=nRemaining * duration)
                #print('Remaining time: ' + str(remainder) + '\n')

          
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
        if (x0!=None and z0!=None):
            return self.field[util.findNearest(self.x, x0),util.findNearest(self.z,z0)]
        if (x0!=None and z0==None): 
            return self.field[util.findNearest(self.x, x0),:]     
        if (x0==None and z0!=None):
            return self.field[:,util.findNearest(self.z,z0)]
        return self.field
                                   

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
        dIndex = int( round((depth + self.filterDepth + self.airHeight) / self.dz) )
        
        return vec[dIndex]
    
    def backwards_solver(self, rxList= np.array([]), freq_min = 0, freq_max = 0, nDiv= 1, R_threshold = 0.1):
        # Cut over your frequency space
        #freq_cut = util.cut_xaxis(self.freq, freq_min, freq_max)

        if self.freqNum > 1:
            idx_min = util.findNearest(freq_min, self.freq)
            idx_max = util.findNearest(freq_max, self.freq)
        else:
            idx_min = 0
            idx_max = 1
        #nFreq = len(freq_cut)

        ### check for Receivers ###
        if (len(rxList) == 0):
            print("Warning: Running time-domain simulation with no receivers. Field will not be saved.")
            #pass
        for rx in rxList:
            rx.setup(self.freq, self.dt)

        for iFreq in range(idx_min, idx_max):
            if iFreq % nDiv == 0:
                tstart_i = time.time()  # Start a time for every frequency step
                freq_i = self.freq[iFreq]  # Frequency_i

                #print('solving for: f = ', freq_i, 'GHz, A = ', self.A[iFreq], 'step:', iFreq - idx_min, 'steps left:', idx_max - iFreq)

                # Add U_positive field
                u_plus = 2 * self.A[iFreq] * self.source * self.filt * freq_i  # Set Forward Propogating Field u_plus
                self.field[0, :] = u_plus[self.fNum:-self.fNum]
                alpha = np.exp(1.j * self.dx * self.k0[iFreq] * (np.sqrt(1. - (self.kz/self.k0[iFreq])**2) - 1.))

                for jXstep in range(1, self.xNum):
                    n_j = self.n2[jXstep, :]  # Verticle referactive index profile at x_i, n_i(z) = n(x = x_i, z)
                    x_j = self.x[jXstep]

                    B = n_j ** 2 - 1
                    Y = np.sqrt(1. + (n_j / self.n0) ** 2)
                    beta = np.exp(1.j * self.dx * self.k0[iFreq] * (np.sqrt(B + Y ** 2) - Y))

                    u_plus = alpha * (util.doFFT(u_plus))
                    u_plus = beta * (util.doIFFT(u_plus))
                    u_plus = self.filt * u_plus

                    self.field[jXstep, :] = (u_plus[self.fNum:-self.fNum] / np.sqrt(x_j)) * np.exp(1.j * self.k0[iFreq] * x_j)

                    #Calculate Reflections range-wise
                    dNx = n_j - self.n2[jXstep - 1, :]
                    reflelction_z = util.reflection_coefficient(n_j, self.n2[jXstep - 1, :])  # Calculate Reflection coefficient

                    if any(reflelction_z) > R_threshold: #TODO -> Should this be reflection_z ** 2?
                        refl_field = np.zeros((jXstep+1, self.zNum), dtype='complex')

                        refl_source = np.zeros(self.zNumFull, dtype='complex')
                        refl_source[self.fNum:-self.fNum] = self.field[jXstep, :] * reflelction_z[self.fNum:-self.fNum]
                        #Scale forward going reduced field by transmission coefficient (has to be smaller than last one)
                        u_plus[self.fNum:-self.fNum] *= util.transmission_coefficient(n_j, self.n2[jXstep - 1, :])[self.fNum:-self.fNum] #TODO -> field or u_plus??

                        #Create Negative Travelling Reduced Field
                        #print(len(refl_source), len(self.filt) )
                        u_minus = 2 * refl_source * self.filt * self.freq[iFreq]
                        refl_field[jXstep, :] = u_minus[self.fNum:-self.fNum]

                        nSteps_backwards = jXstep - 1  # Number of Steps back to origin x=0
                        mXstep = jXstep - 1  # This indice goes backwards

                        alpha_minus = np.exp(1.j * self.dx * self.k0[iFreq] * (np.sqrt(1. - (self.kz/ self.k0[iFreq])**2) - 1.))
                        for kBack in range(nSteps_backwards):
                            n_k = self.n2[mXstep, :]
                            x_minus = abs(self.x[mXstep] - x_j)

                            B_minus = n_k ** 2 - 1
                            Y_minus = np.sqrt(1. + (n_k / self.n0) ** 2)
                            beta_minus = np.exp(1.j * self.dx * self.k0[iFreq] * (np.sqrt(B_minus + Y_minus ** 2) - Y_minus))

                            u_minus = alpha_minus * (util.doFFT(u_minus))
                            u_minus = beta_minus * (util.doIFFT(u_minus))
                            u_minus = self.filt * u_minus

                            refl_field[mXstep, :] = (u_minus[self.fNum:-self.fNum] / np.sqrt(x_minus)) * np.exp(1j*x_minus*self.k0[iFreq])
                            mXstep -= 1

                        self.field[:jXstep, :] += refl_field[:jXstep, :]#TODO -> Should I be adding field components or reduced field components??
                if len(rxList) > 0:
                    for rx in rxList:
                        rx.add_spectrum_component(self.freq[iFreq], self.get_field(x0=rx.x, z0=rx.z))
                tend_i = time.time()
                duration = tend_i - tstart_i
                #print('Solution complete, time: ', round(duration,2), 's')
                nRemaining = (idx_max-iFreq)/nDiv
                #print('Remaining Iterations', nRemaining)
                remainder = datetime.timedelta(seconds = nRemaining*duration)
                #print('Remaining time: ' + str(remainder) + '\n')

    def backwards_solver_2way(self, rxList = np.array([]), freq_min = 0, freq_max = 1, nDiv=1, R_threshold=0.01):
        #New method for calculating backwards waves using u_minus -> use a 3D array to hold reflection sources
        # Cut over your frequency space
        # freq_cut = util.cut_xaxis(self.freq, freq_min, freq_max)
        if self.freqNum > 1:
            idx_min = util.findNearest(freq_min, self.freq)
            idx_max = util.findNearest(freq_max, self.freq)
        else:
            idx_min = 0
            idx_max = 1
        # nFreq = len(freq_cut)

        ### check for Receivers ###
        if (len(rxList) == 0):
            #pass
            print("Warning: Running time-domain simulation with no receivers. Field will not be saved.")
        for rx in rxList:
            rx.setup(self.freq, self.dt)
        for iFreq in range(idx_min, idx_max):
            if iFreq % nDiv == 0:
                tstart_i = time.time()  # Start a time for every frequency step
                freq_i = self.freq[iFreq]  # Frequency_i

                #print('solving for: f = ', freq_i, 'GHz, A = ', self.A[iFreq], 'step:', iFreq - idx_min, 'steps left:', idx_max - iFreq)

                # Add U_positive field
                u_plus = 2 * self.A[iFreq] * self.source * self.filt * freq_i  # Set Forward Propogating Field u_plus
                self.field[0, :] = u_plus[self.fNum:-self.fNum]
                alpha = np.exp(1.j * self.dx * self.k0[iFreq] * (np.sqrt(1. - (self.kz/self.k0[iFreq])**2) - 1.))

                #Backwards Reflection Source
                #refl_source_3arr = np.zeros((self.xNum, self.zNum), dtype='complex')
                refl_source_list = [] #list that contains the reflection sources
                nRefl = 0

                #Solve for u_plus -> from x = 0, x = R
                time_plus_l = []
                time_minus_l = []
                time_xtotal_l = []
                tstart_i = time.time()
                for jXstep in range(1, self.xNum):
                    tstart_xplus = time.time()

                    n_j = self.n2[jXstep, :]  # Verticle referactive index profile at x_i, n_i(z) = n(x = x_i, z)
                    x_j = self.x[jXstep]

                    B = n_j ** 2 - 1
                    Y = np.sqrt(1. + (n_j / self.n0) ** 2)
                    #print('shape k0', self.k0[iFreq].shape, 'shape B', B.shape, 'shape Y', Y.shape)
                    beta = np.exp(1.j * self.dx * self.k0[iFreq] * (np.sqrt(B + Y ** 2) - Y))

                    u_plus = alpha * (util.doFFT(u_plus))
                    u_plus = beta * (util.doIFFT(u_plus))
                    u_plus = self.filt * u_plus

                    self.field[jXstep, :] = (u_plus[self.fNum:-self.fNum] / np.sqrt(x_j)) * np.exp(1.j * self.k0[iFreq] * x_j)
                    tend_xplus = time.time()
                    time_plus_l.append(tend_xplus - tstart_xplus)

                    # Calculate Reflections range-wise
                    dNx = n_j.real - (self.n2[jXstep - 1, :]).real
                    reflelction_z = util.reflection_coefficient(n_j.real, (self.n2[jXstep - 1,:]).real)  # Calculate Reflection coefficient

                    if any(reflelction_z) > R_threshold:
                        nRefl += 1
                        refl_source = np.zeros(self.zNumFull, dtype='complex')
                        refl_source[self.fNum:-self.fNum] = self.field[jXstep, :] * reflelction_z[self.fNum:-self.fNum]

                        refl_field = np.zeros((self.xNum, self.zNum), dtype='complex')
                        refl_field[jXstep,:] = refl_source[self.fNum:-self.fNum]
                        refl_source_list.append(refl_field)
                        # Scale forward going reduced field by transmission coefficient (has to be smaller than last one)
                        u_plus[self.fNum:-self.fNum] *= util.transmission_coefficient(n_j, self.n2[jXstep - 1, :])[self.fNum:-self.fNum]  # TODO -> field or u_plus??

                #Complete forward propagation
                #Commence backwards propagation
                #print('Number of reflections encountered: ', nRefl)
                if nRefl > 0:
                    refl_source_3arr = np.zeros((self.xNum, self.zNumFull, nRefl), dtype='complex')
                    for k in range(nRefl):
                        refl_source_3arr[:,self.fNum:-self.fNum,k] = refl_source_list[k]

                    mXstep = self.xNum - 1
                    u_minus = np.zeros((nRefl, self.zNumFull), dtype='complex')
                    alpha_minus = np.exp(1.j * self.dx * self.k0[iFreq] * (np.sqrt(1. - (self.kz / self.k0[iFreq]) ** 2) - 1.))
                    refl_field_3arr = np.zeros((self.xNum, self.zNum, nRefl))

                    #print('zNum',self.zNum, 'zNumFull (including filtered depths', self.zNumFull)
                    for kBack in range(1, self.xNum): #Make j steps backwards
                        tstart_xminus = time.time()
                        if refl_source_3arr[kBack].any() > 0:
                            filt2 = np.array([self.filt]*nRefl)
                            u_minus[:,:] += 2 * np.transpose(refl_source_3arr[mXstep,:,:]) * filt2 * self.freq[iFreq]

                        n_k = np.array([self.n2[mXstep, :]]*nRefl)
                        x_minus = abs(self.iceLength - self.x[mXstep])

                        B_minus = n_k ** 2 - 1
                        #n0_k = np.array([self.n0]*nRefl)
                        Y_minus = np.sqrt(1. + (n_k / self.n0) ** 2)
                        #k0_k = np.array([self.k0]*nRefl)

                        #print(k0_k.shape, B_minus.shape, Y_minus.shape)
                        #beta_minus = np.exp(1.j * self.dx * k0_k[iFreq] * (np.sqrt(B_minus + Y_minus ** 2) - Y_minus))
                        beta_minus = np.exp(1.j * self.dx * self.k0[iFreq] * (np.sqrt(B_minus + Y_minus ** 2) - Y_minus))
                        filt_k = np.array([self.filt]*nRefl)

                        #TODO: Check if FFT can operate on a 2D array?
                        u_minus = alpha_minus * (util.doFFT(u_minus))
                        #print(u_minus.shape)
                        u_minus = beta_minus * (util.doIFFT(u_minus))
                        #print(u_minus.shape)
                        u_minus = filt_k * u_minus
                        #print(u_minus.shape)

                        mXstep -= 1
                        #refl_field_3arr[mXstep, :, :] = np.transpose((u_minus[:, self.fNum:-self.fNum] / np.sqrt(x_minus)) * np.exp(1j * x_minus * k0_k))
                        refl_field_3arr[mXstep, :, :] = np.transpose(
                            (u_minus[:, self.fNum:-self.fNum] / np.sqrt(x_minus)) * np.exp(1j * x_minus * self.k0[iFreq]))
                        tend_xminus = time.time()
                        time_minus_l.append(tend_xminus - tstart_xminus)
                for k in range(nRefl):
                    self.field[:,:] += refl_field_3arr[:,:,k]
                tend_i = time.time()
                #print('time per pos x step', np.mean(time_plus_l))
                #print('time per negative x step', np.mean(time_minus_l))
                #print('simulation per frequency step', tend_i - tstart_i, 'time per x step (average)', (tend_i-tstart_i)/self.xNum)
                if len(rxList) > 0:
                    for rx in rxList:
                        rx.add_spectrum_component(self.freq[iFreq], self.get_field(x0=rx.x, z0=rx.z))


    def get_backwards_field(self, rxList = np.array([]), freq_min = 0, freq_max = 1, nDiv=1, R_threshold=0.1):
        #New method for calculating backwards waves using u_minus -> use a 3D array to hold reflection sources
        # Cut over your frequency space
        # freq_cut = util.cut_xaxis(self.freq, freq_min, freq_max)
        if self.freqNum > 1:
            idx_min = util.findNearest(freq_min, self.freq)
            idx_max = util.findNearest(freq_max, self.freq)
        else:
            idx_min = 0
            idx_max = 1
        # nFreq = len(freq_cut)
        field_backwards = np.zeros((self.xNum, self.zNum), dtype='complex')

        ### check for Receivers ###
        if (len(rxList) == 0):
            #pass
            print("Warning: Running time-domain simulation with no receivers. Field will not be saved.")
        for rx in rxList:
            rx.setup(self.freq, self.dt)
        for iFreq in range(idx_min, idx_max):
            if iFreq % nDiv == 0:
                tstart_i = time.time()  # Start a time for every frequency step
                freq_i = self.freq[iFreq]  # Frequency_i

                #print('solving for: f = ', freq_i, 'GHz, A = ', self.A[iFreq], 'step:', iFreq - idx_min, 'steps left:', idx_max - iFreq)

                # Add U_positive field
                u_plus = 2 * self.A[iFreq] * self.source * self.filt * freq_i  # Set Forward Propogating Field u_plus
                self.field[0, :] = u_plus[self.fNum:-self.fNum]
                alpha = np.exp(1.j * self.dx * self.k0[iFreq] * (np.sqrt(1. - (self.kz/self.k0[iFreq])**2) - 1.))

                #Backwards Reflection Source
                #refl_source_3arr = np.zeros((self.xNum, self.zNum), dtype='complex')
                refl_source_list = [] #list that contains the reflection sources
                nRefl = 0

                #Solve for u_plus -> from x = 0, x = R
                time_plus_l = []
                time_minus_l = []
                time_xtotal_l = []
                tstart_i = time.time()
                for jXstep in range(1, self.xNum):
                    tstart_xplus = time.time()

                    n_j = self.n2[jXstep, :]  # Verticle referactive index profile at x_i, n_i(z) = n(x = x_i, z)
                    x_j = self.x[jXstep]

                    B = n_j ** 2 - 1
                    Y = np.sqrt(1. + (n_j / self.n0) ** 2)
                    beta = np.exp(1.j * self.dx * self.k0[iFreq] * (np.sqrt(B + Y ** 2) - Y))

                    u_plus = alpha * (util.doFFT(u_plus))
                    u_plus = beta * (util.doIFFT(u_plus))
                    u_plus = self.filt * u_plus

                    self.field[jXstep, :] = (u_plus[self.fNum:-self.fNum] / np.sqrt(x_j)) * np.exp(1.j * self.k0[iFreq] * x_j)
                    tend_xplus = time.time()
                    time_plus_l.append(tend_xplus - tstart_xplus)

                    # Calculate Reflections range-wise
                    dNx = n_j - self.n2[jXstep - 1, :]
                    reflelction_z = util.reflection_coefficient(n_j, self.n2[jXstep - 1,:])  # Calculate Reflection coefficient

                    if any(reflelction_z) > R_threshold:
                        nRefl += 1
                        refl_source = np.zeros(self.zNumFull, dtype='complex')
                        refl_source[self.fNum:-self.fNum] = self.field[jXstep, :] * reflelction_z[self.fNum:-self.fNum]

                        refl_field = np.zeros((self.xNum, self.zNum), dtype='complex')
                        refl_field[jXstep,:] = refl_source[self.fNum:-self.fNum]
                        refl_source_list.append(refl_field)
                        # Scale forward going reduced field by transmission coefficient (has to be smaller than last one)
                        u_plus[self.fNum:-self.fNum] *= util.transmission_coefficient(n_j, self.n2[jXstep - 1, :])[self.fNum:-self.fNum]  # TODO -> field or u_plus??

                #Complete forward propagation
                #Commence backwards propagation
                #print('Number of reflections encountered: ', nRefl)
                if nRefl > 0:
                    refl_source_3arr = np.zeros((self.xNum, self.zNumFull, nRefl), dtype='complex')
                    for k in range(nRefl):
                        refl_source_3arr[:,self.fNum:-self.fNum,k] = refl_source_list[k]

                    mXstep = self.xNum - 1
                    u_minus = np.zeros((nRefl, self.zNumFull), dtype='complex')
                    alpha_minus = np.exp(1.j * self.dx * self.k0[iFreq] * (np.sqrt(1. - (self.kz / self.k0[iFreq]) ** 2) - 1.))
                    refl_field_3arr = np.zeros((self.xNum, self.zNum, nRefl))

                    #print('zNum',self.zNum, 'zNumFull (including filtered depths', self.zNumFull)
                    for kBack in range(1, self.xNum): #Make j steps backwards
                        tstart_xminus = time.time()
                        if refl_source_3arr[kBack].any() > 0:
                            filt2 = np.array([self.filt]*nRefl)
                            u_minus[:,:] += 2 * np.transpose(refl_source_3arr[mXstep,:,:]) * filt2 * self.freq[iFreq]

                        n_k = np.array([self.n2[mXstep, :]]*nRefl)
                        x_minus = abs(self.iceLength - self.x[mXstep])

                        B_minus = n_k ** 2 - 1
                        #n0_k = np.array([self.n0]*nRefl)
                        Y_minus = np.sqrt(1. + (n_k / self.n0) ** 2)
                        k0_k = np.array([self.k0]*nRefl)
                        beta_minus = np.exp(1.j * self.dx * k0_k * (np.sqrt(B_minus + Y_minus ** 2) - Y_minus))

                        filt_k = np.array([self.filt]*nRefl)

                        #TODO: Check if FFT can operate on a 2D array?
                        u_minus = alpha_minus * (util.doFFT(u_minus))
                        #print(u_minus.shape)
                        u_minus = beta_minus * (util.doIFFT(u_minus))
                        #print(u_minus.shape)
                        u_minus = filt_k * u_minus
                        #print(u_minus.shape)

                        mXstep -= 1
                        refl_field_3arr[mXstep, :, :] = np.transpose((u_minus[:, self.fNum:-self.fNum] / np.sqrt(x_minus)) * np.exp(1j * x_minus * k0_k))
                        tend_xminus = time.time()
                        time_minus_l.append(tend_xminus - tstart_xminus)
                for k in range(nRefl):
                    field_backwards += refl_field_3arr[:,:,k]
                    self.field[:,:] += refl_field_3arr[:,:,k]
                tend_i = time.time()
                if len(rxList) > 0:
                    for rx in rxList:
                        rx.add_spectrum_component(self.freq[iFreq], self.get_field(x0=rx.x, z0=rx.z))
        return field_backwards


