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
from types import ModuleType, MethodType
from typing import Any
import os
import webbrowser
import numpy as np
from numpy.typing import NDArray
from ceffyl import models
from ceffyl.priors import Uniform
from ceffyl.ptadata import PTAData

# if available, import jax and numpyro for probabilistic programming
try:
    from jax.scipy.stats import uniform as jax_uniform, norm as jax_norm
    from numpyro.distributions import (Uniform as NumpyroUniform,
                                       Normal as NumpyroNormal)
except ImportError:
    pass


class Spectrum:
    """
    A class to define a spectral model for a pulsar timing array signal. The
    class is designed to model a common red process (e.g. a GW signal) or an
    individual red process (e.g. intrinsic red noise). The class is designed to
    be used in the Ceffyl class to fit noise spectra to a KDE representation of
    an MCMC posterior from a PTA analysis. The class contains methods to
    calculate the logpdf of proposed parameters given the priors,
    to calculate the PSD of proposed parameters from a given PSD function,
    and to sample from the priors.

    Args:

    """
    def __init__(
        self,
        ptadata: PTAData,
        nfreqs: int,
        psd: ModuleType = models.powerlaw,
        priors: list = None,
        const_params: dict = None,
        common_process: bool = True,
        name: str = 'gw',
        psd_kwargs: dict = None,
        selected_psrs: list[str] = None,
        df: float = None,
    ):
        """
        Initialise a signal class to model intrinsic red noise or a common
        process

        Args:
            n_freqs:
                Number of frequencies for this signal. Expected to be equal
                or less than the number of frequencies used to preproces data.
                This fits the first n_freqs frequencies to the data
            freq_idxs:
                an array of indices of frequencies to fit to data. This
                is an alternative to n_freqs. e.g. if you'd like
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
        self.ptadata = ptadata
        self.nfreqs = nfreqs
        self.psd = psd
        self.const_params = const_params if const_params is not None else {}
        self.common_process = common_process
        self.name = name
        self.psd_kwargs = psd_kwargs if psd_kwargs is not None else {}

        if df is None:
            self.df = 1 / ptadata.tspan

        # set default parameters if none given
        if priors is None:
            priors = [Uniform(-18, -12, name='log10_A'),
                      Uniform(0, 7, name='gamma')]
        self.priors = priors

        if selected_psrs is None:
            selected_psrs = ptadata.pulsar_names
        self.selected_psrs = selected_psrs

        # save this information if signal is common to all pulsars
        if common_process:
            param_names = []
            for p in priors:
                if p.size is None or p.size == 1:
                    param_names.append(f'{name}_{p.name}')
                else:
                    param_names.extend([f'{name}_{p.name}_{ii}'
                                        for ii in range(p.size)])
            self.param_names = param_names
            self.n_params = len(param_names)
            self.priors = priors

            # tuple to reshape xs for vectorised computation
            self.reshape = (1, len(priors))

        # else save this information if signal is not common
        # it essentially multiplies lists across psrs for easy mapping
        else:        
            param_names = []
            for p in priors:
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
                                selected_psrs for p in priors]

            self.n_params = len(self.param_names)
            self.priors = priors * len(selected_psrs)
            self.length = len(self.priors)

            # tuple to reshape xs for vectorised computation
            self.reshape = (len(selected_psrs), len(priors))

        self.cp_signals, self.red_signals = [], []

    def get_logpdf(self, xs):
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
        return np.sum([p.logpdf(x) for p, x in zip(self.priors, xs)])

    def get_rho(self,
                freqs: NDArray,
                mapped_xs: dict[str, Any]
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
        
        rho = self.psd(freqs, self.df, **mapped_xs, **self.const_params,
                       **self.psd_kwargs)

        return rho

    def sample(self) -> NDArray:
        """
        Method to derive a list of samples from varied parameters

        Returns
        -------
        samples : NDArray
            array of samples from each parameter
        """
        return np.hstack([p.sample() for p in self.priors])


class Ceffyl:
    """
    A class to fit signals to free spectra to derive the signals' spectral
    characteristics via sampling methods
    """
    def __init__(self,
                 ptadata: PTAData | list[PTAData],
                 spectra: list[Spectrum],
                 ):
        """
        Initialise the Ceffyl object
        
        Args:
            pulsar_list: A list of Pulsar objects
        """
        if isinstance(ptadata, PTAData):
            self.ptadata = [ptadata]
        else:
            self.ptadata = ptadata
        
        self.spectra = spectra
        self.cp_signals, self.red_signals = [], []
        for s in spectra:
            if s.common_process:
                self.cp_signals.append(s)
            else:
                self.red_signals.append(s)
        
        pulsar_names = set()
        for s in spectra:
            pulsar_names.update(s.selected_psrs)
        self.pulsar_names = list(pulsar_names)

        idx = 0
        for s in spectra:
            pmap = []
            if s.common_process:
                for p in s.priors:
                    if p.size is None or p.size == 1:
                        pmap.append(list(np.arange(idx, idx+1)))
                        idx += 1
                    else:
                        pmap.append(list(np.arange(idx, idx+p.size)))
                        idx += p.size
                s.pmap = pmap

            else:
                id_irn = idx
                for ii, p in enumerate(s.priors):
                    if p.size is None or p.size == 1:
                        pmap.append(list(np.arange(id_irn+ii,
                                                   id_irn+s.n_params,
                                                   s.length)))
                    else:
                        if len(s.priors) > 1:
                            raise TypeError("Sorry, ceffyl can't manage more" +
                                            " than one parameter if a " +
                                            "parameter has size > 1")
                        else:
                            npsr = len(s.selected_psrs)
                            array = np.arange(id_irn+ii, id_irn+npsr*p.size)
                            pmap.extend(list(array.reshape(npsr, p.size)))

                if p.size is None or p.size == 1:
                    idx += s.n_params
                else:
                    idx += npsr * p.size

                s.pmap = pmap
        
        # create a list of sample mappings for vectorised searching
        mapped_xs = []
        for s in spectra:
            s.ixgrid = np.ix_(s.selected_psrs, np.arange(s.nfreqs))
            mapped_xs.append({s_i.name: s.params[p] for p, s_i in
                              zip(s.pmap, s.priors)})
        self.mapped_xs = mapped_xs

        self._I, self._J = np.ogrid[:len(self.pulsar_names),
                                    :max([s.nfreqs for s in spectra])]


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
        for s in self.spectra:  # iterate through signals
            mxs = self.mapped_xs[s]
            logpdf += np.sum([p.logpdf(x) for p, x in zip(s.priors, xs[mxs])])

        return logpdf
    
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

        transformed_priors = []
        for s in self.spectra:  # iterate through signals
            mxs = self.mapped_xs[s]
            transformed_priors.extend(
                [p.ppf(x) for p, x in zip(s.priors, xs[mxs])]
                )

        return transformed_priors
    
    def ln_likelihood(self, xs: NDArray) -> float:
        """
        log likelihood function for PTMCMC to calculate logpdf of
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
        logpdf : float
            total logpdf of proposed values given KDE density array
        """

        # initalise array of rho values
        rho = np.zeros((len(self.pulsar_names), max([s.nfreqs for s in self.spectra])))
        for s in self.spectra:  # iterate through signals
            # reshape array to vectorise to size (N_kwargs, N_sig_psrs)
            mxs = self.mapped_xs[s]
            if s.common_process:
                mapped_xs = {s_i.name: xs[mxs] for s_i in s.priors}
            else:
                mapped_xs = {s_i.name: xs[mxs].reshape(s.reshape) for s_i in s.priors}
            rho[s.ixgrid] += s.get_rho(s.ptadata.freqs[s.freq_idxs], mapped_xs=mapped_xs)

        logrho = 0.5*np.log10(rho)  # calculate log10 root PSD

        # search for location of logrho values within grid
        idx = np.searchsorted(self.ptadata.binedges, logrho) - 1

        idx[idx < 0] = 0  # if spectrum less than logrho, set to bottom boundary

        # create a mask to temporarily deal with spectra greater than top
        # boundary
        mask = idx >= self.ptadata.rho_grid.shape[0]
        idx[mask] = -1  # set spectra greater than top boundary to top boundary for now

        logpdf = self.ptadata.log_densities[self._I, self._J, idx]  # logpdf of rho values

        # upper boundary of grid is set to small value
        logpdf[mask] = -36.

        logpdf += np.log(self.ptadata.grid_delta)  # integration infinitessimal
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
