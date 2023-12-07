# imports
import numpy as np
from enterprise.signals.parameter import Uniform
from ceffyl import models
import os
import webbrowser

"""
Classes to create noise signals and a parallel tempered PTMCMCSampler object
to fit spectra to a density estimation of pulsar timing array data
"""


class signal():
    """
    A class to add signals to the GFL
    These signals can be a common process (GW) or individual to each pulsar
    (intrinsic red noise)
    """
    def __init__(self, N_freqs=None, freq_idxs=None, selected_psrs=None,
                 psd=models.powerlaw, params=[Uniform(-18, -12)('log10_A'),
                                              Uniform(0, 7)('gamma')],
                 const_params={}, common_process=True, name='gw_',
                 psd_kwargs={}):
        """
        Initialise a signal class to model intrinsic red noise or a common
        process

        //Inputs//
        @param N_freqs: Number of frequencies for this signal. Expected to be
                        equal or less than the number of frequencies used to
                        preprocess data. This fits the first N_freqs
                        frequencies to the data

        @param freq_idxs: an array of indices of frequencies to fit to data.
                          This is an alternative to N_freqs. e.g. if you'd like
                          to fit data to every second frequency, input
                          freq_idxs=[0,2,4,6,...]

        @param selected_psrs: A list of names of the pulsars under this signal.
                              Expected to be a subset of pulsars within density
                              array loaded in GFL class

        @param psd: A function from enterprise.signals.gp_priors to model PSD
                    for given set of frquencies and spectral characteristics

        @param param: A list of parameters from enterprise.signals.gp_priors to
                      vary. Parameters initialised with prior limits, prior
                      distributions, and name corresponding to kwargs in psd

        @param const_params: A dictionary of values to keep constant. Dictonary
                             keys are kwargs for psd, values are floats

        @param common_process: Is this a common process (e.g. GW signal) or not
                               (e.g. instrinsic pulsar red noise)?

        @param name: What do you want to call your signal? If you're using
                      multiple signals, change this name each time!

        @param psd_kwargs: A dictionary of kwargs for your selected PSD
                           function)

        """

        # saving class information as properties
        if N_freqs is not None or freq_idxs is not None:
            if N_freqs is not None:
                self.N_freqs = N_freqs
                self.freq_idxs = np.arange(N_freqs)
            else:
                self.freq_idxs = np.array(freq_idxs)
                self.N_freqs = len(freq_idxs)

        else:
            print("Please give me some frequencies to search over...")
            return

        self.psd = psd
        self.psd_priors = params
        self.N_priors = len(params)
        self.const_params = const_params
        self.psd_kwargs = psd_kwargs
        self.psr_idx = []  # to be save later in GFL class

        # save information if signal is common to all pulsars
        if common_process:
            self.CP = True
            self.selected_psrs = selected_psrs

            param_names = []
            for p in params:
                if p.size is None or p.size == 1:
                    param_names.append(f'{name}{p.name}')
                else:
                    param_names.extend([f'{name}{p.name}_{ii}'
                                        for ii in range(p.size)])

            self.param_names = param_names
            self.N_params = len(param_names)
            self.params = params
            # tuple to reshape xs for vectorised computation
            self.reshape = (1, 1, len(params))
            self.length = len(params)

        # else save this information if signal is not common
        # it essentially multiplies lists across psrs for easy mapping
        else:
            self.CP = False
            self.selected_psrs = selected_psrs
            self.N_psrs = len(selected_psrs)

            """
            for p in params:
                if p.size is not None:
                    print('single pulsars with varying parameters for each ' +
                          'frequency is not yet supported')
                    return
                else:
                    size = 1
            """

            param_names = []
            for p in params:
                if p.size is None or p.size == 1:
                    param_names.extend([f'{q}_{name}_{p.name}' for q in
                                        selected_psrs])
                else:
                    param_names.extend([f'{q}_{name}_{p.name}_{ii}'
                                        for q in selected_psrs
                                        for ii in range(p.size)])

            self.param_names = param_names

            self.param_names = [f'{name}{q}_{p.name}' for q in
                                selected_psrs for p in params]

            self.N_params = len(self.param_names)
            self.params = params*self.N_psrs
            # tuple to reshape xs for vectorised computation
            self.reshape = (1, len(selected_psrs),
                            len(params))
            self.length = len(self.params)

    def get_logpdf(self, xs):
        """
        A method to calculate total logpdf of proposed values

        //Input//
        @param xs: array of proposed values corresponding to signal params
                   Size=(number of kwargs, number of psrs)

        @return logpdf: summed logpdf of proposed parameter
        """
        return sum([p.get_logpdf(x)  # require 2 x sum of list of arrays
                    for p, x in zip(self.params, xs)]).sum().sum()

    def get_rho(self, freqs, mapped_xs, Tspan):
        """
        A method to calculate PSD of proposed values from given psd function

        //Input//
        @param freqs: Array of PTA frequencies. NOTE: function expects number
                      of frequencies to be greater than or equal to number
                      of frequencies specified for this signal (N_freqs)

        @param mapped_xs: mapped dictionary of proposed values corresponding to
                          signal params

        @return rho: array of PSDs in shape (N_p x N_f)
        """
        rho = self.psd(freqs, Tspan, **mapped_xs, **self.const_params,
                       **self.psd_kwargs).T

        return rho

    def sample(self):
        """
        Method to derive a list of samples from varied parameters

        @return samples: array of samples from each parameter
        """
        return np.hstack([p.sample() for p in self.params])


class ceffyl():
    """
    // Ceffyl //

    A class to fit signals to free spectra to derive the signals' spectral
    characteristics
    """
    def __init__(self, datadir, pulsar_list=None, hist=False, Tspan=None):
        """
        Initialise the class and return a ceffyl object

        @params signals: A list of signals to be searched over
        @params datadir: location of directory containing numpy arrays of
                         densities, corresponding log10rho grids and labels,
                         chain names, and frequencies
        @params pulsar_list: list of pulsars to search over
        @param hist: Flag to state that you're using histograms instead of
                     KDEs
        @param Tspan: Manually enter a Tspan. Default = Tspan of preprocessed
                      data. I recommend keeping it to default

        @param ceffyl: return a ceffyl object
        """

        # checking if datadir exists
        # returns an error if not
        if not os.path.isdir(datadir):
            print("oops! not a directory!")
            webbrowser.open('https://youtu.be/iVSrEn561Bc?t=17')
            raise FileNotFoundError

        # saving properties
        self.freqs = np.load(f'{datadir}/freqs.npy')
        self.N_freqs = self.freqs.size
        self.reshaped_freqs = self.freqs.reshape((1, self.N_freqs)).T
        if Tspan is None:
            self.Tspan = 1/self.freqs[0]
        else:
            self.Tspan = Tspan
        self.rho_labels = np.loadtxt(f'{datadir}/log10rholabels.txt',
                                     dtype=np.unicode_, ndmin=1)
        if hist:
            self.binedges = np.load(f'{datadir}/binedges.npy',
                                    allow_pickle=True)
        else:
            rho_grid = np.load(f'{datadir}/log10rhogrid.npy')

            db = rho_grid[1] - rho_grid[0]
            binedges = rho_grid - 0.5*db
            binedges = np.append(binedges, binedges[-1]+0.5*db)

            self.rho_grid, self.binedges, self.db = rho_grid, binedges, db

        # selected pulsars
        if pulsar_list is None:
            self.pulsar_list = list(np.loadtxt(f'{datadir}/pulsar_list.txt',
                                               dtype=np.unicode_, ndmin=1))
        else:
            self.pulsar_list = pulsar_list
        self.N_psrs = len(self.pulsar_list)

        # find index of sublist
        file_psrs = list(np.loadtxt(f'{datadir}/pulsar_list.txt',
                                    dtype=np.unicode_,
                                    ndmin=1))
        selected_psrs = [file_psrs.index(p) for p in self.pulsar_list]

        # load densities from npy binary file for given psrs, freqs
        density_file = f'{datadir}/density.npy'
        density = np.load(density_file, allow_pickle=True)[selected_psrs]

        self.density = np.nan_to_num(density, nan=-36.)

        return

    def add_signals(self, signals, inverse_transform=False,
                    nested_posterior_sample_size=10000):
        """
        Method to add signals to the ceffyl object

        @param nested_posterior_sample_size: number of sample to setup
                                             posterior histograms for nested
                                             sampling
        """

        # check if signals is a list
        if not isinstance(signals, list):
            raise TypeError("Please supply of signals as a list")

        # set number of freqs to max number of freqs of signals
        # self.N_freqs = max([s.N_freqs for s in signals])

        # check if pulsars in signals are in density array
        for s in signals:
            if s.selected_psrs is None:
                s.selected_psrs = self.pulsar_list

            self.N_psrs = len(s.selected_psrs)  # save number of psrs

            if not np.isin(s.selected_psrs, self.pulsar_list).all():
                raise ValueError('Mismatch between density array pulsars and' +
                                 'the pulsars you selected')

            else:  # save idx of (subset of) psrs within larger list
                s.psr_idx = np.array([list(self.pulsar_list).index(p)
                                      for p in s.selected_psrs])

        # precomputing parameter locations in proposed arrays
        id = 0
        self.cp_signals, self.red_signals = [], []
        for s in signals:
            pmap = []
            if s.CP:
                self.cp_signals.append(s)
                for p in s.params:
                    if p.size is None or p.size == 1:
                        pmap.append(list(np.arange(id, id+1)))
                        id += 1
                    else:
                        pmap.append(list(np.arange(id, id+p.size)))
                        id += p.size
                s.pmap = pmap

            else:
                self.red_signals.append(s)
                id_irn = id
                for ii, p in enumerate(s.psd_priors):
                    if p.size is None or p.size == 1:
                        pmap.append(list(np.arange(id_irn+ii,
                                                   id_irn+s.N_params,
                                                   s.N_priors)))
                    else:
                        if len(s.psd_priors) > 1:
                            print("Sorry, ceffyl can't manage more than one" +
                                  " parameter if a parameter has size > 1")
                            raise TypeError
                        else:
                            npsr = len(s.selected_psrs)
                            array = np.arange(id_irn+ii, id_irn+npsr*p.size)
                            pmap.extend(list(array.reshape(npsr, p.size)))

                if p.size is None or p.size == 1:
                    id += s.N_params
                else:
                    id += npsr * p.size

                s.pmap = pmap

        # create list of idx grids
        for s in signals:
            s.ixgrid = np.ix_(s.psr_idx, s.freq_idxs)

        # save array of signals
        self.signals = signals

        # save complete array of parameters
        self.param_names = list(np.hstack([s.param_names for s in signals]))
        self.params = list(np.hstack([s.params for s in signals]))
        self.ndim = len(self.param_names)

        # setup empty 2d grid to vectorize product of pdfs
        self._I, self._J = np.ogrid[:self.N_psrs, :self.N_freqs]

        # information for nested sampling
        if inverse_transform:
            posterior_samples, hist_cumulative, binmid = [], [], []
            for s in self.signals:  # iterate through signals
                # if binmid is None:
                for ii, p in enumerate(s.params):
                    if p.size is None or p.size == 1:
                        posterior_samples = [s.psd_priors[ii].sample()
                                             for jj in
                                             range(nested_posterior_sample_size)]
                        hist, bin_edges = np.histogram(posterior_samples,
                                                       bins='fd')
                        hist_cumulative.append(np.cumsum(hist/hist.sum()))
                        binmid.append((bin_edges[:-1] + bin_edges[1:])/2)
                    else:
                        # FIX ME: nested sampling for free spec irn
                        print('Free spectrum not supported with nested sampling yet!')
                        hist_cumulative, binmid = None, None

            self.hist_cumulative = hist_cumulative
            self.binmid = binmid

        return self

    def ln_prior(self, xs):
        """
        vectorised log prior function for PTMCMC to calculate logpdf of
        proposed values given their parameter distribution

        @param xs: proposed value array

        @return logpdf: total logpdf of proposed values given signal parameters
        """
        logpdf = 0  # total logpdf
        for s in self.signals:  # iterate through signals
            # reshape array to vectorise to size (N_kwargs, N_sig_psrs)
            mapped_x = [xs[p] for p in s.pmap]
            logpdf += s.get_logpdf(mapped_x)

        return logpdf

    def transform_uniform(self, u):
        """
        prior function for using in nested samplers

        it transforms the N-dimensional unit cube u to our prior range of
        interest

        NOTE: assumes uniform priors

        @param u: N-dimensional unit cube
        @return x: transformed prior
        """

        x = u.copy()  # copy hypercube

        for s in self.signals:  # iterate through signals
            for ii, p in enumerate(s.pmap):
                prior_min = s.psd_priors[ii].prior._defaults['pmin']
                prior_max = s.psd_priors[ii].prior._defaults['pmax']
                prior_diff = prior_max - prior_min

                x[p] = x[p]*prior_diff + prior_min

        return x

    def transform_histogram(self, xs):
        """
        prior function for using in nested samplers

        it transforms the N-dimensional unit cube u to our prior range of
        interest

        tutorial for this function:
        https://johannesbuchner.github.io/UltraNest/priors.html#Non-analytic-priors

        @param u: N-dimensional unit cube
        @return x: transformed prior
        """
        x = np.zeros_like(xs)
        for s in self.signals:  # iterate through signals
            for ii, p in enumerate(s.pmap):
                x[p] = np.interp(xs[p], self.hist_cumulative[ii],
                                 self.binmid[ii])

        return x
    
    def hypercube(self, xs):
        """
        function to compute ppf of the prior to use in nested sampling
        REQUIRES: enterprise fork by vhaasteren:
        git@github.com:vhaasteren/enterprise.git

        @param xs: proposed value array
        """

        transformed_priors = np.empty_like(xs)  # initialise empty array

        for ii, p in enumerate(self.params):  # iterate through signals
            transformed_priors[ii] = p.ppf(xs[ii])

        return transformed_priors

    def ln_likelihood(self, xs, jacobian=False):
        """
        vectorised log likelihood function for PTMCMC to calculate logpdf of
        proposed values given KDE density array

        @param xs: proposed value array

        @return logpdf: total logpdf of proposed values given KDE density array
        """

        # initalise array of rho values with lower prior boundary
        red_rho, cp_rho = np.zeros((self.N_psrs, self.N_freqs)), np.zeros((self.N_psrs, self.N_freqs))
        for s in self.red_signals:  # iterate through signals
            # reshape array to vectorise to size (N_kwargs, N_sig_psrs)
            mapped_xs = {s_i.name: xs[p]
                         for p, s_i in zip(s.pmap, s.params)}
            red_rho[s.ixgrid] += s.get_rho(self.reshaped_freqs[s.freq_idxs],
                                           Tspan=self.Tspan,
                                           mapped_xs=mapped_xs)

        for s in self.cp_signals:  # iterate through CP signals
            # reshape array to vectorise to size (N_kwargs, N_sig_psrs)
            mapped_xs = {s_i.name: xs[p]
                         for p, s_i in zip(s.pmap, s.params)}
            cp_rho[s.ixgrid] += s.get_rho(self.reshaped_freqs[s.freq_idxs],
                                          Tspan=self.Tspan,
                                          mapped_xs=mapped_xs)

        rho = red_rho + cp_rho  # total rho
        logrho = 0.5*np.log10(rho)  # calculate log10 root PSD

        # search for location of logrho values within grid
        idx = np.searchsorted(self.binedges, logrho) - 1

        if (idx >= self.rho_grid.shape[0]).any():
            return -np.inf
        
        idx[idx < 0] = 0  # if spectrum less than logrho, set to bottom boundary

        logpdf = self.density[self._I, self._J, idx]  # logpdf of rho values

        if jacobian:  # flag to test jacobian requirement
            # compute the jacobian
            jac = np.sqrt((red_rho**2 + cp_rho**2)) / rho
            logpdf += np.log(jac)  # is there a factor of two mistake?

        logpdf += np.log(self.db)  # integration infinitessimal
        return np.sum(logpdf)

    """
    def hist_ln_likelihood(self, xs):
        #
        log likelihood function for PTMCMC to calculate logpdf of
        proposed values given histogram density arrays. This isn't optimised
        for speed. It is best for PTA freespec only

        @param xs: proposed value array

        @return logpdf: total logpdf of proposed values given KDE density array
        #
        rho = np.zeros((self.N_psrs, self.N_freqs))  # initalise empty array
        for s in self.signals:  # iterate through signals
            # reshape array to vectorise to size (N_kwargs, N_sig_psrs)
            mapped_x = {s_i.name: xs[p][:, None]
                        for p, s_i in zip(s.pmap, s.psd_priors)}
            rho[s.psr_idx,
                :s.N_freqs] += s.get_rho(self.reshaped_freqs[:s.N_freqs],
                                         mapped_x)

        logrho = 0.5*np.log10(rho)  # calculate log10 root PSD

        # search for location of logrho values within grid and logpdf
        logpdf = 0
        for ii in range(self.N_psrs):
            for jj in range(self.N_freqs):
                idx = np.searchsorted(self.binedges[ii*self.N_freqs+jj],
                                      logrho[ii][jj]) - 1
                logpdf += self.density[ii][jj][idx]

        return logpdf
    """

    def initial_samples(self):
        """
        A method to return an array of initial random samples for PTMCMC

        @return x0: array of initial samples
        """
        x0 = np.hstack([s.sample() for s in self.signals])
        return x0
