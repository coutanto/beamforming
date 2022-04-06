#
#
# Module to compute beamforming, largely taken from covnet package by L. Seydoux
# ported to gpu by O. Coutant using cupy
# additional option adapted to DAS data by allowing a diagonal selection on cross-spectral matrix
#
#

import numpy as np

column_width = [28, 48]
char_corner = '-'
char_vline = ' '
char_line = '-'
char_wait = '|'

CMPLX_GPU='complex64'
FLOAT_GPU='float32'

__doc__="Module to compute beamforming using classical or music projection, largely inspired by covnet module by L. Seydoux, gpu porting by 0. Coutant"

class Beamformer():
    def __init__(self):
        self.n_slowness = None # number of slowness
        self.slowness = None   # list of slowness values
        self.phase_x = None    # phase shift of the reference wave
        self.phase_y = None
        self.beamformer = None  # multicolumn vector describing the reference wave we project on
        self.beamformer_conj = None # conjugate version of above
        self.id = 'BF'

class Plane_wave_beamformer(Beamformer):
    def  __init__(self):
        super().__init__()
        self.slowness_grid = None # slowness grid as a meshgrid
        self.slowness_abs = None
        self.X_radial = None #coeff. used to performed radial/transverse beamforming on 2 components data
        self.Y_radial = None
        self.X_tang = None
        self.Y_tang = None
        self.id = 'PWBF'

class Cylindrical_wave_beamformer(Beamformer):
    def __init__(self):
        super().__init__()
        self.x0 = None # grid for source position
        self.y0 = None
        self.id = 'CWBF'
        self.n_source = None


class Beam:
    def __init__(self, use_gpu=False):
        """
        ## Description
            Create a Beam instance needed to compute the beamforming

        ## Input:
            use_gpu: (bool) if True, attempt to use cupy

        ## Example:
            >>> import numpy as np
            >>> x = np.array(list_of_positions_x)
            >>> y = np.array(list_of_positions_y)
            >>> beam = a1das.beamforming.Beam(use_gpu=True)
            >>> beam.set_receiver(x=x, y=y)
            >>> f=a1das.open(file)
            >>> a1 = f.read()
            >>> beam.set_data(a1section = a1)
            >>> times = beam.compute_crossspectral_matrix(duration,
                                              freq_min,
                                              freq_max)
            >>> freq = 10 #Hz
            >>> vmin = 0.1 #km/sec
            >>> freq_id = beam.compute_planewave_beamformer(freq,
                                                slowness_max=1./vmin)
            >>> time_id = 0 #first time index in times
            >>> beam.compute_beam_projection(time_id,freq_id)
            >>> beam.pcolormesh(plt.gca(), cmap="RdYlBu_r")
        """
        if use_gpu:
            try:
                import cupy
                print('module cupy found, using gpu')
                self.use_gpu = True
            except:
                print('gpu requested but cupy module is not available, pip install cupy ?')
                self.use_gpu = False
        else:
            self.use_gpu = False
        
        
        # original data
        self.x = None           # x & y position of stations
        self.y = None
        self.nstat = None       # number of stations
        self.data = None        # one component data
        self.data_X = None      # 2 components data
        self.data_Y = None
        self.dt = None      # time step
        self.time = None    #sample time of original data
        
        # parameter of analysis
        self.frequencies = None # list of frequencies for analysis
        self.xspec = None   # 4D array of cross-spectra [time, freq, nstat, nstat]
        self.wtime = None   # time of moving windows
        self.wlen = None    # duration (sec) of a moving window

        self.bf = Beamformer() # reference wave wo project on (plane, cylindrical, etc)
        self.beam = None    # result of beamforming


        self.time_id = None #current time index used for beamforming, refering to self.wtime



    def __str__(self):
        """
        """
        print('Beam object has the following members:\n')
        for key in self.__dict__.keys():
            print(key)
    def __repr__(self):
        """
        """
        print('Beam object has the following members:\n')
        for key in self.__dict__.keys():
            print(key)



# =============================================================================
# =============================================================================
#        set_receiver
# =============================================================================
# =============================================================================
    def set_receiver(self, **kwargs):
        """
        ## Description:
            Set position parameters of stations

        ## Input:
            x, y: (float) ndarray of size [nstat]
            or
            xy: (float) ndarray of size [nstat x 2]
        """
        from numpy import asarray
        if self.use_gpu:
            import cupy as cp
            FLOAT = FLOAT_GPU
        else:
            import numpy as cp
            FLOAT = 'float64'

        if 'xy' in kwargs:
            xy = kwargs['xy']
            self.x = asarray(xy[0, :], dtype=FLOAT)
            self.y = asarray(xy[1, :], dtype=FLOAT)
        elif 'x' in kwargs and 'y' in kwargs:
            self.x = asarray(kwargs['x'], dtype=FLOAT)
            self.y = asarray(kwargs['y'], dtype=FLOAT)
        else:
            raise ValueError('missing input arguments xy or x and y')
        self.nstat = len(self.x)

# =============================================================================
# =============================================================================
#        set_XY_data
# =============================================================================
# =============================================================================
    def set_XY_data(self, data_X, data_Y, fs, trange=None):
        """
        ## Description:
            set horizontal 2 components data

        ## Input:
            data_X, data_Y: (float) 2D ndarray of size (nstat, ntime) containing seismic data
            fs: (float) sampling rate
            trange: (int) list or tuple (index_tim_min, index_time_max)
        """
        if trange is not None:
            tstart, tend = trange
        else:
            tstart = 0;
            tend = -1
        if data_X.shape != data_Y.shape:
            raise ValueError(' data_X and data_Y dimension mismatch')
        if self.data is not None:
            raise ValueError('You already set 1 component dataset')


        self.data_X = data_X[tstart:tend]
        self.data_Y = data_Y[tstart:tend]
        self.dt = 1./fs
        self.ntime = self.data.shape[1]
        self.ntrace = self.data.shape[0]
        self.time = np.linspace(0, self.ntime) * self.dt
# =============================================================================
# =============================================================================
#           set_data
# =============================================================================
# =============================================================================
    def set_data(self, drange=None, ddecim=1, trange=None, **kwargs):
        """
        ## Description:
        Set seismic data

        ## Input from DAS data using A1Section object:
            a1section: A1Section object
            drange: [dmin, dmax] (float) tuple or list of distance range (default = None)
            ddecim: (int) space decimation
            trange: [tmin, tmax] (float) tuple or list of time range (default = None)

        ## Input from float array
            section: (float) 2D ndarray of size nstat x npts
            trange: (int) list or tuple (index_tim_min, index_time_max)
            fs: (float) sampling rate
        """
        if self.data_X is not None or self.data_Y is not None:
            raise ValueError('You already set 2 components dataset')

        if 'a1section' in kwargs:
            from a1das.core import A1Section
            a1 = kwargs['a1section']
            if drange is not None:
                dstart,dend = A1Section.index(drange=drange)
            else:
                dstart=None; dend=None
            if trange is not None:
                tstart, tend = A1Section.index(trange=trange)
            else:
                tstart=None; tend=None
            self.data = a1.data[dstart:dend:ddecim,tstart:tend]
            self.dt = a1['dt']
            self.time = a1['time'][tstart:tend]
            self.ntime = len(self.time)
            self.ntrace = self.data.shape[0]

        elif 'section' in kwargs and 'fs' in kwargs:
            if trange is not None:
                tstart, tend = trange
            else:
                tstart=None; tend=None
            self.data = kwargs['section'][:,tstart:tend]
            self.dt = 1./kwargs['fs']
            self.ntime = self.data.shape[1]
            self.ntrace = self.data.shape[0]
            self.time = np.arange(0,self.ntime)*self.dt

        else:
            raise ValueError("wrong input args, a1section or section AND dt")
            
# =============================================================================
# =============================================================================
#           compute_crossspectral_matrix
# =============================================================================
# =============================================================================

    def compute_crossspectral_matrix(self, segment_duration_sec, freq_min, freq_max, 
                                     step = 1., average=10, overlap=0.5, mx=None):
        """
        ## Description
            Compute the cross-spectral matrix on moving time window

        ## Input for FFT computations
            segment_duration: (float) time window length for analysis
            step: (float) time step between two consecutive windows

        ## Input for cross-spectral computations
            freq_min: (float)
            freq_max: (float)
            average:  (int) cross spectrum computed over 'average' time windows
            overlap: (int) overlap between consecutive average is 'overlap' time window
            mx: (int) compute xspec on the mx first neighbors only, default=None, compute cross-spectrum on all neighbors

        Note: numpy code source inspired or copied from covnet by leonard Seydoux
        """
        from scipy.signal.windows  import hann
        from scipy.linalg import circulant
        from logtable import waitbar

        if self.use_gpu:
            import cupy as cp
            CMPLX = CMPLX_GPU
            data = self.data.astype('float32')
            has_cupy = True
        else:
            import numpy as cp
            CMPLX = 'complex128'
            data = self.data
            has_cupy = False

        # Time windows
        # set length argument for fft call
        # force even length
        len_seg = int(segment_duration_sec / self.dt)
        if np.mod(len_seg,2) == 1:
            len_seg = 2*int(np.floor(len_seg/2)+1)
        nfft = len_seg
        len_step = int(np.floor(len_seg * step))
        times = self.time[:1 - len_seg:len_step]
        n_windows = len(times)
        self.wlen = len_seg*self.dt

        # Frequency
        n_frequencies = int(nfft/2+1) #see rfft doc
        df = 1./(self.dt*nfft)
        self.frequency = np.arange(0, n_frequencies)*df

        # =========================================================================
        # compute spectra
        # =========================================================================
        spectra_shape = self.ntrace, n_windows, n_frequencies
        spectra = cp.zeros(spectra_shape, dtype=CMPLX)
        wbar = waitbar('Spectra', self.ntrace)
        data_gpu = cp.array(data)
        hanning_gpu = cp.array(hann(len_seg))
        for trace_id in range(0, self.ntrace):
            wbar.progress(trace_id)
            tr = data_gpu[trace_id]
            for time_id in range(n_windows):
                start = time_id * len_step
                end = start + len_seg
                segment = tr[start:end] * hanning_gpu
                spectra[trace_id, time_id] = cp.fft.rfft(segment, n=nfft)
                
        #free unused memory in GPU, i.e. original data
        if has_cupy:
            data_gpu = None
            hanning_gpu = None
            tr = None
            segment = None
            cp.get_default_memory_pool().free_all_blocks()


        # add a supplementary time step for later averaging
        t_end = self.time[-1]
        times = np.hstack((times, t_end))

        # =========================================================================
        # compute cross-spectral matrix
        # =========================================================================

        # select frequencies
        freq_keep = (self.frequency > freq_min) & (self.frequency < freq_max)
        self.frequency = self.frequency[freq_keep]
        n_frequencies = len(self.frequency)
        spectra = spectra[..., freq_keep].copy()

        # 
        # spectra are average over time if requested
        # set time averaging parameters
        # 
        if average>n_windows:
            average = n_windows
        overlap = int(overlap)

        # Times ??
        t_end = times[-1]
        times = times[:-1]
        times = times[:1 - average:overlap]
        #self.wlen *= average
        n_average = len(times)
        #times = np.hstack((times, t_end))

        # introduce a spatial extent for correlation computation
        # this has a sense only when the network has a 1D regular spacing
        if mx is not None:
            c = np.zeros((self.ntrace,))
            nx = int((mx-1)/2)
            c[0:nx+1]=1.
            c[-nx:]=1.
            cc = cp.asarray(circulant(c))
            cc = cc[:,:,None]
        else:
            cc=1.

        # Initialization
        xspec_shape = (n_average, self.ntrace, self.ntrace, n_frequencies)
        xspec = cp.zeros(xspec_shape, dtype=CMPLX)
        wbar = waitbar('Cross-spectra', n_average)
        for wid in range(n_average):
            #n_traces, n_windows, n_frequencies = spectra.shape
            beg = overlap * wid
            end = beg + average
            spectra_tmp = spectra[:, beg:end, :].copy()

            X = (spectra_tmp[:, None, 0, :] * cp.conj(spectra_tmp[:, 0, :]))*cc
            for swid in range(1, average):
                X += (spectra_tmp[:, None, swid, :] * cp.conj(spectra_tmp[:, swid, :]))*cc

            xspec[wid] = X
            wbar.progress(wid)

        self.xspec = xspec.transpose([0, -1, 1, 2]).copy() #is copy() really necessary?

        if has_cupy:
            spectra_tmp = None
            spectra = None
            X = None
            cc = None
            xspec = None
            cp.get_default_memory_pool().free_all_blocks()

        self.wtime = times
        
# =============================================================================
# =============================================================================
#           compute_XY_crossspectral_matrix
# =============================================================================
# =============================================================================
    def compute_XY_crossspectral_matrix(self, segment_duration_sec, freq_min, freq_max, step = 1., average=10, overlap=0.5, mx=None):
        """
        ## Description
            Compute the cross-spectral matrix on moving time window

        ## Input for FFT computations
            segment_duration: (float) time window length for analysis
            step: (float) time step between two consecutive windows

        ## Input for cross-spectral computations
            freq_min: (float)
            freq_max: (float)
            average:  (int) cross spectrum computed over 'average' time windows
            overlap: (float)
            mx: (int) compute xspec on the mx first neighbors only, default=None, compute cross-spectrum on all neighbors

        Note: numpy code source inspired or copied from covnet by leonard Seydoux
        """
        from scipy.signal import hanning
        from scipy.linalg import circulant
        from .logtable import waitbar

        if self.use_gpu:
            import cupy as cp
            CMPLX = CMPLX_GPU
            data = self.data.astype('float32')
            has_cupy = True
        else:
            import numpy as cp
            CMPLX = 'complex128'
            data = self.data
            has_cupy = False

        # Time windows
        len_seg = int(segment_duration_sec / self.dt)
        len_step = int(np.floor(len_seg * step))
        times = self.time[:1 - len_seg:len_step]
        n_windows = len(times)

        # Frequency
        # set length argument for fft call
        # force even length
        nfft = 2*int(np.floor(len_seg/2)+1)
        len_seg = nfft
        n_frequencies = int(nfft/2+1) #see rfft doc
        df = 1./(self.dt*nfft)
        self.frequency = np.arange(0, n_frequencies)*df

        # =========================================================================
        # compute spectra
        # =========================================================================
        spectra_shape = self.ntrace, n_windows, n_frequencies
        spectra = cp.zeros(spectra_shape, dtype=CMPLX)
        wbar = waitbar('Spectra', self.ntrace)
        self.xspec=()
        for i,data in enumerate(self.data_X, self.data_Y):
            data_gpu = cp.array(data)
            hanning_gpu = cp.array(hanning(len_seg))
            for trace_id in range(0, self.ntrace):
                wbar.progress(trace_id)
                tr = data_gpu[trace_id]
                for time_id in range(n_windows):
                    start = time_id * len_step
                    end = start + len_seg
                    segment = tr[start:end] * hanning_gpu
                    spectra[trace_id, time_id] = cp.fft.rfft(segment, n=nfft)
            #free unused memory in GPU, i.e. original data
            if has_cupy:
                data_gpu = None
                hanning_gpu = None
                tr = None
                segment = None
                cp.get_default_memory_pool().free_all_blocks()


            # add a supplementary time step for loop ending
            t_end = self.time[-1]
            times = np.hstack((times, t_end))

            # =========================================================================
            # compute cross-spectral matrix
            # =========================================================================

            # select frequencies
            freq_keep = (self.frequency > freq_min) & (self.frequency < freq_max)
            self.frequency = self.frequency[freq_keep]
            n_frequencies = len(self.frequency)
            spectra = spectra[..., freq_keep].copy()

            # set time averaging parameters
            overlap = int(average * overlap)

            # Times ??
            t_end = times[-1]
            times = times[:-1]
            times = times[:1 - average:overlap]
            n_average = len(times)
            #times = np.hstack((times, t_end))

            # introduce a spatial extent for correlation computation
            # this has a sense only when the network has a 1D/2D regular grid
            if mx is not None:
                c = np.zeros((self.ntrace,))
                nx = int((mx-1)/2)
                c[0:nx+1]=1.
                c[-nx:]=1.
                cc = cp.asarray(circulant(c))
                cc = cc[:,:,None]
            else:
                cc=1.

            # Initialization
            xspec_shape = (n_average, self.ntrace, self.ntrace, n_frequencies)
            xspec = cp.zeros(xspec_shape, dtype=CMPLX)
            wbar = waitbar('Covariance', n_average)
            for wid in range(n_average):
                #n_traces, n_windows, n_frequencies = spectra.shape
                beg = overlap * wid
                end = beg + average
                spectra_tmp = spectra[:, beg:end, :].copy()

                X = (spectra_tmp[:, None, 0, :] * cp.conj(spectra_tmp[:, 0, :]))*cc
                for swid in range(1, average):
                    X += (spectra_tmp[:, None, swid, :] * cp.conj(spectra_tmp[:, swid, :]))*cc

                xspec[wid] = X
                wbar.progress(wid)

            self.xspec.append(xspec.transpose([0, -1, 1, 2]).copy()) #is copy() really necessary?

        if has_cupy:
            spectra_tmp = None
            spectra = None
            X = None
            cc = None
            xspec = None
            cp.get_default_memory_pool().free_all_blocks()

        self.wtime = times
# =============================================================================
# =============================================================================
#           compute_planewave_beamformer
# =============================================================================
# =============================================================================

    def compute_pw_beamformer(self, frequency, slowness_max=0.1, dimension=100, flip = True,
                                     clean_gpu=False):
        """
        ## Description:
            Compute the set of slowness and delay that will be used for beam projection
            Assume plane wave propagation: exp(1j*2*pi*freq*(Sx*x + Sy*y))
        ## Input:
            frequency: (float) frequency in Hz
            slowness_max: (float) plane wave is computed for Sx and Sy slowness varying between [-slowness_max,slowness_max]
            dimension: (int) number of slowness values in the range
            flip: (bool) flip phase (default = True)
        """
        from numpy import abs
        if self.use_gpu:
            import cupy as cp
            CMPLX = CMPLX_GPU
        else:
            import numpy as cp
            CMPLX = 'complex128'

        id = "PWBF %.1f %d" % (slowness_max, dimension)

        #compute phase shift only if needed
        if id != self.bf.id:
            self.id = id
            self.bf.n_slowness = dimension
            self.bf.slowness_max = slowness_max
            self.bf.slowness = cp.linspace(-slowness_max, slowness_max, dimension)
            self.bf.slowness_grid = cp.meshgrid(self.bf.slowness, self.bf.slowness)
            self.bf.slowness_abs = cp.sqrt(self.bf.slowness_grid[0].ravel()**2 + self.bf.slowness_grid[1].ravel()**2)

            # outer product of (u,v) = dot(u,transpose(v))
            # ravel: return a 1D vector from a 2D matrix
            # return matrix of Sx*x
            self.bf.phase_x = cp.outer(self.bf.slowness_grid[0].ravel(), cp.array(self.x))
            # return matrix of Sy*y
            self.bf.phase_y = cp.outer(self.bf.slowness_grid[1].ravel(), cp.array(self.y))

            # case of 2 component beamforming
            if self.data_X is not None and self.data_Y is not None:
                self.bf.X_radial = self.bf.slowness_grid[0].ravel()/self.bf.slowness_abs
                self.bf.Y_radial = self.bf.slowness_grid[1].ravel()/self.bf.slowness_abs
                self.bf.X_tang = self.bf.Y_radial
                self.bf.Y_tang = -self.bf.X_radial

        freq_id = abs(self.frequency - frequency).argmin()
        frequency = self.frequency[freq_id]

        angular_frequency = 2 * np.pi * frequency
        if flip:
            angular_frequency *= -1

        self.bf.beamformer = cp.exp(1j * angular_frequency * (self.bf.phase_x + self.bf.phase_y)).astype(CMPLX)
        self.bf.beamformer_conj = self.bf.beamformer**(-1)

        #if has_cupy and clean_gpu:
        #    x = None
        #    y = None
        #    phase_x = None
        #    phase_y = None
        #    cp.get_default_memory_pool().free_all_blocks()

        return freq_id

    # =============================================================================
    # =============================================================================
    #           compute_cw_beamformer
    # =============================================================================
    # =============================================================================

    def compute_cw_beamformer(self, x0, y0, frequency, slowness, flip=True, use_meshgrid=False):
        """
        ## Description:
            Compute the set of slowness and delay that will be used for beam projection
            Assume plane wave propagation: exp(1j*2*pi*freq*(Sx*x + Sy*y))
        ## Input:
            x0,y:0 (float) source position in the same unit as position
            frequency: (float) frequency in Hz
            slowness: (float) ndarray or one value,
            dimension: (int) number of slowness values in the range
            flip: (bool) flip phase (default = True)
        """

        from numpy import abs, asarray, ndarray, meshgrid, outer, exp, sqrt
        if self.use_gpu:
            import cupy as cp
            FLOAT = FLOAT_GPU
            CMPLX = CMPLX_GPU
        else:
            import numpy as cp
            FLOAT = 'float32'
            CMPLX = 'complex128'

        id = 'CWBF'
        # compute phase shift only if needed
        self.bf = Cylindrical_wave_beamformer()

        self.bf.slowness = asarray(slowness)
        self.bf.n_slowness = len(slowness)
        if use_meshgrid:
            x0y0 = meshgrid(asarray(x0), asarray(y0))
            x0 = x0y0[0].ravel()
            y0 = x0y0[1].ravel()
            self.bf.x0 = x0
            self.bf.y0 = y0
        else:
            self.bf.x0 = x0
            self.bf.y0 = y0
        self.bf.n_source = len(x0)

        # outer product of (u,v) = dot(u,transpose(v))
        # ravel: return a 1D vector from a 2D matrix
        # return matrix of Sx*x
        phase_x = ndarray((len(x0), len(self.x)), dtype=FLOAT)
        phase_y = ndarray((len(y0), len(self.y)), dtype=FLOAT)
        for i, xx0 in enumerate(x0):
            delta_x = xx0-self.x
            delta_y = y0[i]-self.y
            dist = sqrt(delta_x**2 +delta_y**2)
            nx = delta_x/dist
            ny = delta_y/dist
            phase_x[i,:] = delta_x*nx
            phase_y[i,:] = delta_y*ny
        phase_x = phase_x.ravel()
        phase_y = phase_y.ravel()

        self.bf.phase_x = outer(self.bf.slowness, phase_x)
        self.bf.phase_y = outer(self.bf.slowness, phase_y)

        freq_id = abs(self.frequency - frequency).argmin()
        frequency = self.frequency[freq_id]

        angular_frequency = 2 * np.pi * frequency
        if flip:
            angular_frequency *= -1
        beamformer = exp(1j * angular_frequency * (self.bf.phase_x + self.bf.phase_y)).astype(CMPLX)
        self.bf.beamformer = cp.asarray(beamformer)
        self.bf.beamformer_conj = self.bf.beamformer ** (-1)

        #self.bf.beamformer = self.bf.beamformer.reshape(len(slowness)*len(x0),len(self.x))
        #self.bf.beamformer_conj = self.bf.beamformer_conj.reshape(len(slowness) * len(x0), len(self.x))
        # if has_cupy and clean_gpu:
        #    x = None
        #    y = None
        #    phase_x = None
        #    phase_y = None
        #    cp.get_default_memory_pool().free_all_blocks()


        return freq_id

 # =============================================================================
# =============================================================================
#           compute_pw_beam_projection
# =============================================================================
# =============================================================================

    def compute_pw_beam_projection(self, time_id=0, freq_id=0, method='classic', epsilon=1e-10, rank=1, stack=False):
        """
        ## Description:
        Compute the projection of the cross-spectral matrix on a reference plane wave beam for a given time window and a given frequency

        ## Input:
            time_id: (int) time index for the cross-spectral matrix
            freq_id: (int) frequency index for the cross -sptral matrix
            method: (str) 'classic' or 'music' (default is classic)
            epsilon: (float) threshold for music method, default is 1.e-10
            rank: (int) keep rank singular values in svd decomposition
            stack: (bool) stack projection between successive calls (default = False)
        ## Return:
            nothing, Beam.beam is written
        """
        #from .logtable import waitbar

        if self.use_gpu:
            import cupy as cp
            has_cupy = True
            from cupy.linalg import svd, inv
        else:
            import numpy as cp
            has_cupy = False
            from numpy.linalg import svd, inv

        if method == 'classic':
            if has_cupy:
                #print(type(self.xspec),self.xspec.dtype)
                tmp = self.bf.beamformer_conj @ self.xspec[time_id,freq_id] @ self.bf.beamformer.T
                beam = cp.diag(tmp).real
            else:
                beam = cp.ndarray((self.bf.n_slowness**2,))
                #wbar = waitbar('Projection',self.dimension**2)
                for s in range(self.bf.n_slowness**2):
                    beam[s] = (self.bf.beamformer_conj[s, :].dot(self.xspec[time_id,freq_id].dot(self.bf.beamformer[s, :]))).real
                    #wbar.progress(s)

        elif method == 'music':
            eigenvectors, eigenvalues, _ = svd(self.xspec[time_id, freq_id])
            eigenvalues[:rank] = 0.0
            eigenvalues[rank:] = 1.0
            eigenvalues = cp.diag(eigenvalues)
            if has_cupy:
                #eigenvectors = cp.array(eigenvectors)
                xspec = eigenvectors @ eigenvalues @ cp.conj(eigenvectors.T)
                tmp = self.bf.beamformer_conj @ xspec @ self.bf.beamformer.T
                beam = cp.diag(tmp).real + epsilon
                beam = 1 / cp.abs(beam)
            else:
                beam = cp.ndarray((self.bf.n_slowness**2,))
                xspec = eigenvectors @ eigenvalues @ cp.conj(eigenvectors.T)
                for s in range(self.bf.n_slowness ** 2):
                    beam[s] = (self.bf.beamformer_conj[s, :].dot(xspec.dot(self.bf.beamformer[s, :])).real)
                    beam[s] = 1 / cp.abs(beam[s] + epsilon)

        elif method == 'mvdr':
            if has_cupy:
                xspec_inv = inv(self.xspec[time_id,freq_id]+cp.eye(self.ntrace)*0.01)
                tmp = self.bf.beamformer_conj @ xspec_inv @ self.bf.beamformer.T
                beam = cp.diag(tmp).real #+ epsilon
                beam = 1 / cp.abs(beam)
            else:
                beam = cp.ndarray((self.bf.n_slowness**2,))
                xspec_inv = inv(self.xspec[time_id, freq_id]+cp.eye(self.ntrace)*0.01)
                for s in range(self.bf.n_slowness**2):
                    beam[s] = 1./(self.bf.beamformer_conj[s, :].dot(xspec_inv.dot(self.bf.beamformer[s, :]))).real+epsilon

        if stack and self.beam is not None:
            self.beam += beam
        else:
            self.beam = beam

        self.time_id = time_id

    # =============================================================================
    # =============================================================================
    #           compute_cw_beam_projection
    # =============================================================================
    # =============================================================================

    def compute_cw_beam_projection(self, time_id=0, freq_id=0, method='classic', epsilon=1e-10, rank=1,
                                       stack=False):
        if self.use_gpu:
            import cupy as cp
            has_cupy = True
            from cupy.linalg import svd, inv
        else:
            import numpy as cp
            has_cupy = False
            from numpy.linalg import svd, inv

        if method == 'classic':
            if has_cupy:
                beam = cp.zeros((self.bf.n_slowness, self.bf.n_source))
                for s in range(self.bf.n_slowness):
                    LHS = self.bf.beamformer_conj[s].reshape(self.bf.n_source,self.ntrace)
                    RHS = self.bf.beamformer[s].reshape(self.bf.n_source,self.ntrace)
                    beam[s] = cp.diag((LHS @ self.xspec[time_id, freq_id] @ RHS.T).real)

            else:
                beam = cp.zeros((self.bf.n_slowness,self.bf.n_source))
                # wbar = waitbar('Projection',self.dimension**2)
                for s in range(self.bf.n_slowness):
                    LHS = self.bf.beamformer_conj[s].reshape(self.bf.n_source,self.ntrace)
                    RHS = self.bf.beamformer[s].reshape(self.bf.n_source,self.ntrace)
                    for src in range(self.bf.n_source):
                        beam[s,src] = (LHS[src] @ self.xspec[time_id, freq_id] @ RHS[src]).real
                    
                    #beam[s] = cp.diag((LHS @ self.xspec[time_id, freq_id] @ RHS.T).real)
                    # wbar.progress(s)

        if stack and self.beam is not None:
            self.beam += beam
        else:
            self.beam = beam

        self.time_id = time_id

    # =============================================================================
# =============================================================================
#           compute_XY_beam_projection
# =============================================================================
# =============================================================================
    def compute_pw_XY_beam_projection(self, time_id=0, freq_id=0, method='classic', comp='radial', epsilon=1e-10, rank=1, stack=False):
        """
        ## Description:
        Compute the projection of the cross-spectral matrix on a reference beam for a given time window and a given frequency

        ## Input:
            time_id: (int) time index for the cross-spectral matrix
            freq_id: (int) frequency index for the cross -sptral matrix
            method: (str) 'classic' or 'music' (default is classic)
            epsilon: (float) threshold for music method, default is 1.e-10
            rank: (int) keep rank singular values in svd decomposition
            stack: (bool) stack projection between successive calls (default = False)
        ## Return:
            nothing, Beam.beam is written
        """
        if self.use_gpu:
            import cupy as cp
            has_cupy = True
            from cupy.linalg import svd
        else:
            import numpy as cp
            has_cupy = False
            from numpy.linalg import svd

        if comp == 'radial':
            coefs = self.X_radial, self.Y_radial
        else:
            coefs = self.X_tang, self.Y_tang

        beam = cp.zeros((self.dimension ** 2,))
        if method == 'classic':
            if has_cupy:
                for i,cof in enumerate(coefs):
                    tmp = (cof * self.beamformer_conj) @ self.xspec[i][time_id,freq_id] @ (cof * self.beamformer).T
                    beam += cp.diag(tmp).real
            else:
                for s in range(self.dimension**2):
                    for i,cof in enumerate(coefs):
                        beam[s] += (
                              (self.beamformer_conj[s, :]*cof[s]).dot(
                               self.xspec[i][time_id,freq_id].dot(self.beamformer[s, :]*cof[s])
                                                                       )
                              ).real


        elif method == 'music':
            beam = cp.zeros((self.dimension ** 2,))
            for i, cof in enumerate(coefs):
                eigenvectors, eigenvalues, _ = svd(self.xspec[i][time_id, freq_id])
                eigenvalues[:rank] = 0.0
                eigenvalues[rank:] = 1.0
                eigenvalues = cp.diag(eigenvalues)
                if has_cupy:
                    #eigenvectors = cp.array(eigenvectors)
                    xspec = eigenvectors @ eigenvalues @ cp.conj(eigenvectors.T)
                    tmp = (cof*self.beamformer_conj) @ xspec @ (cof*self.beamformer.T)
                    beam += 1. / (cp.diag(tmp).real + epsilon)
                else:
                    xspec = eigenvectors @ eigenvalues @ cp.conj(eigenvectors.T)
                    for s in range(self.dimension ** 2):
                        beam[s] = ((cof[s]*self.beamformer_conj[s, :]).dot(xspec.dot(cof[s]*self.beamformer[s, :]))).real
                        beam[s] = 1 / cp.abs(beam[s] + epsilon)



        if stack and self.beam is not None:
            self.beam += beam
        else:
            self.beam = beam
        self.time_id = time_id
# =============================================================================
# =============================================================================
#           usefull stuff
# =============================================================================
# =============================================================================

    def copy(self):
        """

        """
        newbeam = Beam()
        for e in self.__dict__:
            newbeam.__dict__[e] = self.__dict__[e]

        return newbeam

    def clear(self):
        """

        """
        if self.use_gpu:
            try:
                import cupy as cp
                for e in self.__dict__:
                    self.__dict__[e] = None
                cp.get_default_memory_pool().free_all_blocks()
            except:
                pass
            
# =============================================================================
# =============================================================================
#           pw_pcolormesh
# =============================================================================
# =============================================================================
    def pw_pcolormesh(self, ax, colorbar=False, scale=True, **kwargs):
        """
        plot result of plane wave beam forming
        """
        if self.use_gpu:
            has_cupy = True
        else:
            has_cupy = False

        import numpy as np
        import matplotlib.pyplot as plt

        if has_cupy:
            beam = self.beam.get()
            slowness = self.bf.slowness.get()
        else:
            beam=self.beam
            slowness = self.bf.slowness

        beam = np.reshape(beam, (self.bf.n_slowness, self.bf.n_slowness))
        beam = np.rot90(beam, 2)
        if scale:
            beam = (beam - beam.min()) / (beam.max() - beam.min())
            vmin=0.
            vmax=1.
        else:
            vmin=beam.min()**2
            vmax=beam.max()**2
        kwargs = {**kwargs, **dict(rasterized=True, vmin=vmin, vmax=vmax)}
        img = ax.pcolormesh(slowness, slowness, beam ** 2, shading='auto', **kwargs)
        ax.set_aspect('equal')
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        grid_style = dict(lw=0.3, dashes=[6, 4], c='w')
        ax.plot(2 * [0], xlim, **grid_style)
        ax.plot(ylim, 2 * [0], **grid_style)
        ax.plot(xlim, ylim, **grid_style)
        ax.plot(xlim, [-y for y in ylim], **grid_style)
        ax.set_xticks([xlim[0], xlim[0] / 2, 0, xlim[-1] / 2, xlim[-1]])
        ax.set_yticks([ylim[0], ylim[0] / 2, 0, ylim[-1] / 2, ylim[-1]])
        ax.set_title('[%.1f-%.1f]sec' % (self.wtime[self.time_id], 
                                           self.wtime[self.time_id]+self.wlen))
        if colorbar:
            plt.colorbar(mappable=img)

    # =============================================================================
    # =============================================================================
    #           cw_pcolormesh
    # =============================================================================
    # =============================================================================
    def cw_pcolormesh(self, ax, colorbar=False, scale=True, **kwargs):
        """
        plot result of cylindrical wave beam forming
        """
        if self.use_gpu:
            has_cupy = True
        else:
            has_cupy = False

        import numpy as np
        import matplotlib.pyplot as plt

        if has_cupy:
            beam = self.beam.get()
        else:
            beam = self.beam

        beam = np.reshape(beam, (self.bf.n_slowness, self.bf.n_source))
        if scale:
            beam = (beam - beam.min()) / (beam.max() - beam.min())
            vmin = 0.
            vmax = 1.
        else:
            vmin = beam.min() ** 2
            vmax = beam.max() ** 2
        kwargs = {**kwargs, **dict(rasterized=True, vmin=vmin, vmax=vmax)}
        img = ax.pcolormesh(np.asarray(range(0,self.bf.n_source)), self.bf.slowness,
                            beam ** 2, shading='auto', **kwargs)
        ax.set_aspect('auto')
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        grid_style = dict(lw=0.3, dashes=[6, 4], c='w')
        #ax.plot(2 * [0], xlim, **grid_style)
        #ax.plot(ylim, 2 * [0], **grid_style)
        #ax.plot(xlim, ylim, **grid_style)
        #ax.plot(xlim, [-y for y in ylim], **grid_style)
        #ax.set_xticks([xlim[0], xlim[0] / 2, 0, xlim[-1] / 2, xlim[-1]])
        #ax.set_yticks([ylim[0], ylim[0] / 2, 0, ylim[-1] / 2, ylim[-1]])
        plt.xlabel('Index source position')
        plt.ylabel('slowness')
        ax.set_title('[%.1f-%.1f]sec' % (self.wtime[self.time_id],
                                         self.wtime[self.time_id] + self.wlen))
        if colorbar:
            plt.colorbar(mappable=img)

# =============================================================================
# =============================================================================
#           compute_plane_wave_synthetic_crossspectral_matrix
# =============================================================================
# =============================================================================
    def compute_pw_synthetic_crossspectral_matrix(self, freq, slowness, azimuth,mx=None):
        """

        :param freq:
        :param slowness:
        :param azimuth:
        :return:
        """
        if self.use_gpu:
            import cupy as cp
            from scipy.linalg import circulant
        else:
            import numpy as cp
            from scipy.linalg import circulant
            
        # Phase
        wavenumber = 2 * cp.pi * freq * slowness
        azimuth = cp.radians(azimuth)
        scalar_product = cp.sin(azimuth) * self.x + cp.cos(azimuth) * self.y
    
        # Wavefield
        wavefield = cp.exp(-1j * wavenumber * scalar_product)

        #spatial smoothing
        if mx is not None:
            c = np.zeros((self.ntrace,))
            nx = int((mx - 1) / 2)
            c[0:nx + 1] = 1.
            c[-nx:] = 1.
            cc = cp.asarray(circulant(c))
            #cc = cc[:, :, None]
        else:
            cc = 1.
        # cross-spectra
        self.xspec = (wavefield * wavefield.conj()[:, None]) * cc
        self.xspec = self.xspec[None, None, :,:]

    # =============================================================================
    # =============================================================================
    #           compute_cw_synthetic_crossspectral_matrix
    # =============================================================================
    # =============================================================================
    def compute_cw_synthetic_crossspectral_matrix(self, freq, slowness, x0, y0, mx=None):
        """

        :param freq:
        :param slowness:
        :param azimuth:
        :return:
        """


        if self.use_gpu:
            import cupy as cp
            from scipy.linalg import circulant
        else:
            import numpy as cp
        import numpy as np

        # Phase
        angular_frequency = 2 * cp.pi * freq 

        dist = np.sqrt((self.x-x0)**2+(self.y-y0)**2)
        kx = slowness*(self.x-x0)/dist
        ky = slowness*(self.y-y0)/dist
        phase_x = kx*(self.x-x0)
        phase_y = ky*(self.y-y0)
        # Wavefield
        wavefield = np.exp(-1j * angular_frequency * (phase_x+phase_y))

        # spatial smoothing
        if mx is not None:
            c = np.zeros((self.ntrace,))
            nx = int((mx - 1) / 2)
            c[0:nx + 1] = 1.
            c[-nx:] = 1.
            cc = np.asarray(circulant(c))
            # cc = cc[:, :, None]
        else:
            cc = 1.
        # cross-spectra
        self.xspec = np.outer(wavefield,wavefield.conj()) * cc
        self.xspec = self.xspec[None, None, :, :]
        self.xspec = cp.array(self.xspec)







