"""
Classes to create noise spectra and a parallel tempered PTMCMCSampler
object to fit spectral models to a free spectrum of pular timing array data.

The Ceffyl class is a class to fit sources' spectra to a compressed
representation of pulsar timing array data (called a free spectrum). The
class is designed to fit spectra to a free spectrum of a common red process
(with or without inter-pulsar correlations) for an ensemble of pulsars (the
'PTA Free Spectrum'), or to fit individual and/or common red processes to an
ensemble of free spectra of individual pulsars ('GFL Lite'/'GFL').

The Spectrum class is a class to create a noise spectrum for a pulsar timing
array. The class is designed to model a common red process (e.g. a GW signal)
or an individual red process (e.g. intrinsic red noise) for a pulsar timing
array. The class is designed to be used in the Ceffyl class to fit noise
spectra to a free spectrum of pulsar timing array data.

Example:

    # import Ceffyl class
    from ceffyl import Ceffyl

    # create a Ceffyl object
    ceffyl = Ceffyl(datadir='path/to/data')

    # create a Spectrum object
    spectrum = Spectrum(n_freqs=10, selected_psrs=['J1713+0747'])

    # add the Spectrum object to the Ceffyl object
    ceffyl.add_signals([spectrum])

    # run the PTMCMCSampler
    ceffyl.run_sampler()

    # plot the results
    ceffyl.plot_results()


Classes:

    Ceffyl:
        A class to fit sources' spectra to a compressed representation of
        pulsar timing array data (called a free spectrum).

    Spectrum:
        A class to create a noise spectrum for a pulsar timing array.
"""

# imports
from types import ModuleType
from typing import Any
import os
import webbrowser
import numpy as np
from numpy.typing import NDArray
from ceffyl import models
from ceffyl.priors import Uniform
from ceffyl.pulsar import Pulsar, Frequencies


class Spectrum:
    """
    A class to add spectra to free spectra for fitting in Ceffyl.

    The Spectrum class is a class to create a noise spectrum from a given
    GW source. The class is designed to model a common red process (e.g. a GW
    signal) or an individual red process (e.g. intrinsic red noise) for a
    pulsar timing array. The class is designed to be used in the Ceffyl class
    to fit noise spectra to a free spectrum of pulsar timing array data.

    Args:

    """
    def __init__(
        self,
        psrs: Pulsar | list[Pulsar],
        nfreqs: int,
        tspan: float = None,
        psd: ModuleType = models.powerlaw,
        params: list | np.ndarray = None,
        common_process: bool = True,
        name: str = 'gw_',
        psd_kwargs: dict = None
    ):
        """
        Initialise a signal class to model intrinsic red noise or a common
        process

        Args:
            N_freqs:
                Number of frequencies for this signal. Expected to be equal
                or less than the number of frequencies used to preproces data.
                This fits the first N_freqs frequencies to the data
            freq_idxs:
                an array of indices of frequencies to fit to data. This
                is an alternative to N_freqs. e.g. if you'd like
                          to fit data to every second frequency, input
                          freq_idxs=[0,2,4,6,...]
            selected_psrs:
                A list of names of the pulsars under this signal.
                              Expected to be a subset of pulsars within density
                              array loaded in GFL class
            psd:
                A function from enterprise.signals.gp_priors to model PSD
                    for given set of frquencies and spectral characteristics

            param:
                A list of parameters from enterprise.signals.gp_priors to
                      vary. Parameters initialised with prior limits, prior
                      distributions, and name corresponding to kwargs in psd

            const_params:
                A dictionary of values to keep constant. Dictonary
                             keys are kwargs for psd, values are floats

            common_process:
                Is this a common process (e.g. GW signal) or not
                               (e.g. instrinsic pulsar red noise)?

            name:
                What do you want to call your signal? If you're using
                      multiple signals, change this name each time!

            psd_kwargs:
                A dictionary of kwargs for your selected PSD
                           function)
        """
        # set default parameters if none given
        if params is None:
            params = [Uniform(-18, -12, name='log10_A'),
                      Uniform(0, 7, name='gamma')]

        # saving class information as properties
        self.name = name
        self.common_process = common_process

        self.nfreqs = nfreqs
        if tspan is None:
            tspan = np.zeros(len(psrs))
            for ii, p in enumerate(psrs):
                tspan[ii] = p.tspan

        # saving metadata to class
        self.psd = psd
        self.psd_priors = params
        self.n_priors = len(params)
        self.const_params = const_params
        self.psd_kwargs = psd_kwargs
        self.psr_idx = []  # to be save later in Ceffyl class

        # save this information if signal is common to all pulsars
        if common_process:
            self.cp = True
            self.selected_psrs = selected_psrs

            param_names = []
            for p in params:
                if p.size is None or p.size == 1:
                    param_names.append(f'{name}{p.name}')
                else:
                    param_names.extend([f'{name}{p.name}_{ii}'
                                        for ii in range(p.size)])
            self.param_names = param_names
            self.n_params = len(param_names)
            self.params = params
            self.length = len(params)

            # tuple to reshape xs for vectorised computation
            self.reshape = (1, 1, len(params))

        # else save this information if signal is not common
        # it essentially multiplies lists across psrs for easy mapping
        else:
            self.cp = False
            self.selected_psrs = selected_psrs
            self.n_psrs = len(selected_psrs)

            param_names = []
            for p in params:
                if p.size is None or p.size == 1:
                    param_names.extend(
                        [f'{q}_{name}_{p.name}' for q in selected_psrs]
                        )
                else:
                    param_names.extend(
                        [f'{q}_{name}_{p.name}_{ii}' for q in selected_psrs
                         for ii in range(p.size)]
                         )

            self.param_names = param_names

            self.param_names = [f'{name}{q}_{p.name}' for q in
                                selected_psrs for p in params]

            self.n_params = len(self.param_names)
            self.params = params*self.n_psrs
            # tuple to reshape xs for vectorised computation
            self.reshape = (1, len(selected_psrs),
                            len(params))
            self.length = len(self.params)

            # tuple to reshape xs for vectorised computation
            self.reshape = (1, len(selected_psrs), len(params))

    def get_logpdf(self, xs: NDArray) -> float:
        """
        A method to calculate total logpdf of proposed values

        Parameters
        -----
        xs : NDArray 
            array of parameter values

        Returns
        -------
        logpdf : float
            logpdf of proposed parameters for the given models and parameters
        """
        # require 2 x sum of list of arrays
        return np.sum([p.get_logpdf(x) for p, x in zip(self.params, xs)])

    def get_rho(self,
                freqs: NDArray,
                mapped_xs: dict[str, Any],
                tspan: float
                ) -> NDArray:
        """
        A method to calculate PSD of proposed values from given psd function

        Parameters
        ----------
        freqs : NDArray
            Array of PTA frequencies. NOTE: function assumes number of
            frequencies to be greater than or equal to number of frequencies
            specified for this Signal object (N_freqs)

        mapped_xs : dict[str, Any]
            mapped dictionary of proposed values corresponding to Signal params

        Returns
        -------
        rho : NDArray
            array of PSDs in shape (N_p x N_f)
        """
        rho = self.psd(freqs, tspan, **mapped_xs, **self.const_params,
                       **self.psd_kwargs).T

        return rho

    def sample(self) -> NDArray:
        """
        Method to derive a list of samples from varied parameters

        Returns
        -------
        samples : NDArray
            array of samples from each parameter
        """
        return np.hstack([p.sample() for p in self.params])


class Ceffyl:
    """
    A class to fit signals to free spectra to derive the signals' spectral
    characteristics via sampling methods
    """
    def __init__(self, pulsar_list: list[Pulsar] = None,
                 spectra: list[Spectrum] = None,):
        """
        Initialise the Ceffyl object
        
        Args:
            pulsar_list: A list of Pulsar objects
        """

        # storing frequency metadata
        self.freqs = np.load(f'{datadir}/freqs.npy')
        self.n_freqs = self.freqs.size
        self.reshaped_freqs = self.freqs.reshape((1, self.N_freqs)).T
        if tspan is None:
            self.tspan = 1/self.freqs[0]
        else:
            self.tspan = tspan
        self.rho_labels = np.loadtxt(f'{datadir}/log10rholabels.txt',
                                     dtype=np.unicode_, ndmin=1)

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
        self.n_psrs = len(self.pulsar_list)

        # find index of sublist
        file_psrs = list(np.loadtxt(f'{datadir}/pulsar_list.txt',
                                    dtype=np.unicode_, ndmin=1))
        selected_psrs = [file_psrs.index(p) for p in self.pulsar_list]

        # load densities from npy binary file for given psrs
        density_file = f'{datadir}/density.npy'
        density = np.load(density_file, allow_pickle=True)[selected_psrs]

        self.density = np.nan_to_num(density, nan=-36.)

        self.cp_signals, self.red_signals = [], []

    def add_signals(self,
                    signals: list[MethodType],
                    inverse_transform: bool = False,
                    nested_posterior_sample_size: int = 100000
                    ) -> MethodType:
        """
        Method to add signals to the ceffyl object.
        Note: using this method erases previous signals added to instance!

        TO DO
        -----
            * ensure that adding signals doesn't erase previously added signals
            * intrinsic red noise free spectrum

        Parameters
        ----------
        signals : list[MethodType]
            list of Ceffyl.signal objects
        inverse_transform : bool
            flag to compute inverse transforms for use in nested samplers.
            Will soon be depreciated in favour of computing point-percentile
            functions
        nested_posterior_sample_size : int
            number of samples to setup posterior histograms for nested sampling

        Raises
        ------
        TypeError
            if signals aren't supplied as a list
        ValueError
            if pulsars defined in signal object mismatch pulsars in dataset
        """

        # check if signals is a list
        if not isinstance(signals, list):
            raise TypeError("Please supply of signals as a list")

        # cycle through signals and defined pulsar names
        for s in signals:
            if s.selected_psrs is None:
                s.selected_psrs = self.pulsar_list

            self.n_psrs = len(s.selected_psrs)  # save number of psrs

            if not np.isin(s.selected_psrs, self.pulsar_list).all():
                raise ValueError('Mismatch between density array pulsars and'
                                 'the pulsars you selected')
            else:  # save idx of (subset of) psrs within larger list
                s.psr_idx = np.array([list(self.pulsar_list).index(p)
                                      for p in s.selected_psrs])
            
            self.N_psrs = len(s.selected_psrs)  # save number of psrs

        # precomputing parameter locations in proposed arrays
        idx = 0
        for s in signals:
            pmap = []
            if s.CP:
                self.cp_signals.append(s)
                for p in s.params:
                    if p.size is None or p.size == 1:
                        pmap.append(list(np.arange(idx, idx+1)))
                        idx += 1
                    else:
                        pmap.append(list(np.arange(idx, idx+p.size)))
                        idx += p.size
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
                            raise TypeError("Sorry, ceffyl can't manage more" +
                                            " than one parameter if a " +
                                            "parameter has size > 1")
                        else:
                            npsr = len(s.selected_psrs)
                            array = np.arange(id_irn+ii, id_irn+npsr*p.size)
                            pmap.extend(list(array.reshape(npsr, p.size)))

                if p.size is None or p.size == 1:
                    idx += s.N_params
                else:
                    idx += npsr * p.size

                s.pmap = pmap

        # create list of index grids for vectorised searching
        for s in signals:
            s.ixgrid = np.ix_(s.psr_idx, s.freq_idxs)

        self.signals = signals  # save array of signals

        # save complete array of parameters
        self.param_names = list(np.hstack([s.param_names for s in signals]))
        self.params = list(np.hstack([s.params for s in signals]))
        self.ndim = len(self.param_names)

        # only use max number of frequencies
        # TODO: check if this works when using freq_idxs
        self.N_freqs = max([s.N_freqs for s in signals])

        # setup empty 2d grid to vectorize product of pdfs
        self._I, self._J = np.ogrid[:self.n_psrs, :self.N_freqs]

        # information for nested sampling
        if inverse_transform:
            print('Inverse transform sampling method will soon be depreciated')
            print('Please update your version of enterprise to use PPF')
            posterior_samples, hist_cumulative, binmid = [], [], []
            for s in self.signals:  # iterate through signals
                # if binmid is None:
                for ii, p in enumerate(s.params):
                    if p.size is None or p.size == 1:
                        posterior_samples = [
                            s.psd_priors[ii].sample() for jj in
                            range(nested_posterior_sample_size)
                            ]
                        hist, bin_edges = np.histogram(posterior_samples,
                                                       bins='fd')
                        hist_cumulative.append(np.cumsum(hist/hist.sum()))
                        binmid.append((bin_edges[:-1] + bin_edges[1:])/2)
                    else:
                        # FIX ME: nested sampling for free spec irn
                        print('Free spectrum not supported with nested ' +
                              'sampling yet!')
                        hist_cumulative, binmid = None, None

            self.hist_cumulative = hist_cumulative
            self.binmid = binmid

    def ln_prior(self, xs: NDArray) -> float:
        """
        log prior function for PTMCMC to calculate logpdf of
        proposed parameters given their prior distribution

        Parameters
        ----------
        xs : NDArray
            proposed parameters

        Returns
        -------
        logpdf : float
            summed logpdf of proposed parameters given signal priors
        """
        logpdf = 0  # total logpdf
        for s in self.signals:  # iterate through signals
            # reshape array to vectorise to size (N_kwargs, N_sig_psrs)
            mapped_x = [xs[p] for p in s.pmap]
            logpdf += s.get_logpdf(mapped_x)

        return logpdf

    def transform_histogram(self, xs):
        """
        WILL SOON BE DEPRECIATED
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
    
    def hypercube(self, xs : NDArray) -> NDArray:
        """
        function to compute ppf of the prior to use in nested sampling

        Parameters
        ----------
        xs : NDArray
            proposed parameters

        Returns
        -------
        transformed_priors : NDArray
            array of point-percentile function of parameters given priors
        """
        transformed_priors = np.empty_like(xs)  # initialise empty array
        for ii, p in enumerate(self.params):    # iterate through signals
            transformed_priors[ii] = p.ppf(xs[ii])

        return transformed_priors

    def ln_likelihood(self, xs: NDArray) -> np.float64:
        """
        vectorised log likelihood function for PTMCMC to calculate logpdf of
        proposed values given KDE density array

        TO DO
        -----
            * trapesium rule integration

        Parameters
        ----------
        xs : NDArray
            proposed parameters

        Returns
        -------
        logpdf : np.float64
            total logpdf of proposed values given KDE density array
        """

        # initalise array of rho values with lower prior boundary
        red_rho = np.zeros((self.n_psrs, self.n_freqs))
        cp_rho = np.zeros((self.n_psrs, self.n_freqs))
        for s in self.red_signals:  # iterate through signals
            # reshape array to vectorise to size (N_kwargs, N_sig_psrs)
            mapped_xs = {s_i.name: xs[p]
                         for p, s_i in zip(s.pmap, s.params)}
            red_rho[s.ixgrid] += s.get_rho(self.reshaped_freqs[s.freq_idxs],
                                           Tspan=self.tspan,
                                           mapped_xs=mapped_xs)

        for s in self.cp_signals:  # iterate through CP signals
            # reshape array to vectorise to size (N_kwargs, N_sig_psrs)
            mapped_xs = {s_i.name: xs[p]
                         for p, s_i in zip(s.pmap, s.params)}
            cp_rho[s.ixgrid] += s.get_rho(self.reshaped_freqs[s.freq_idxs],
                                          Tspan=self.tspan,
                                          mapped_xs=mapped_xs)

        rho = red_rho + cp_rho  # total rho
        logrho = 0.5*np.log10(rho)  # calculate log10 root PSD

        # search for location of logrho values within grid
        idx = np.searchsorted(self.binedges, logrho) - 1
    
        idx[idx < 0] = 0  # if spectrum less than logrho, set to bottom boundary

        # if any spectrum greater than top boundary, set logprob to -np.inf
        # else compute probability given data
        if (idx >= self.rho_grid.shape[0]).any():
            return -np.inf

        idx[idx < 0] = 0  # if spectrum less than logrho, set to lower boundary

        logpdf = self.density[self._I, self._J, idx]  # logpdf of rho values

        logpdf += np.log(self.db)  # integration infinitessimal
        return np.sum(logpdf)

    def initial_samples(self) -> NDArray:
        """
        A method to return an array of initial random samples for PTMCMC

        Returns
        -------
        x0 : NDArray
            array of sample parameters
        """
        x0 = np.hstack([s.sample() for s in self.signals])
        return x0
