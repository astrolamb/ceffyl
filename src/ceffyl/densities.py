"""
A module to create density estimators of Bayesian MCMC posteriors for pulsars

This module contains the `DE_Factory` class, which is designed to create
density estimators of Bayesian MCMC posteriors for pulsars. It can handle
multiple pulsars and their respective MCMC chains, compute bandwidths for
the density estimators, and generate kernel density estimates (KDEs) of
the Bayesian posteriors. The module supports both single pulsar analyses
and PTA free spectrum analyses, allowing for flexible handling of pulsar
data. It also includes methods for saving the computed densities and
metadata to files, making it suitable for further analysis or visualization.
"""

import time
from types import MethodType
from typing import Any
import itertools
import warnings
import os
import numpy as np
from emcee.autocorr import integrated_time
from KDEpy import FFTKDE
from numpy.typing import NDArray
from ceffyl import bandwidths as bw
from ceffyl.pulsar import PTAData
try:
    from joblib import Parallel, delayed
    no_joblib = False
except ImportError:
    no_joblib = True
    print('Joblib cannot be found. You cannot setup densities for multiple '
          'pulsars simultaneously.')

class ChainProcessor:
    """
    A class to create density estimations for each MCMC chain and
    save them as an array of probabilities. This is specifically designed to
    work on chains from an MCMC 'free spectrum' analysis from enterprise-pulsar
    (https://github.com/nanograv/enterprise/)
    """
    def __init__(self,
                 chains: NDArray,
                 pulsar_names: list[str] = None,
                 param_labels: list[str] = None,
                 freqs: NDArray = None):
        """
        Open the compressed chain files and create density estimators

        Parameters
        ----------
        chains : NDArray
            MCMC chains to create density estimators for. This is a 2D array
            with shape (N_psrs, N_freqs, N_samples) where N_psrs is the number
            of pulsars, N_freqs is the number of frequencies, and N_samples is
            the number of samples in the chain.
        pulsar_names : list[str]
            list of pulsar names to save
        param_labels : list[str]
            list of parameter labels to save
        freqs : NDArray
            1D array of frequencies corresponding to the MCMC chains.
            This should have the same length as the number of frequencies in
            the chains. If None, it will be set to a default range.

        Raises
        ------
        ValueError
            If the length of `freqs` does not match the number of frequencies
            in `chains`.
        ValueError
            If the shape of `chains` is not 3D or does not match the expected
            dimensions for pulsars, frequencies, and samples.
        ValueError
            If the length of `pulsar_names` does not match the number of
            pulsars in `chains`.
        ValueError
            If the length of `param_labels` does not match the number of
            frequencies in `chains`.

        Notes
        -----
        The `chains` parameter should be a 3D numpy array where the first
        dimension corresponds to pulsars, the second dimension corresponds to
        frequencies, and the third dimension corresponds to samples. The
        `freqs` parameter should be a 1D numpy array of frequencies that matches
        the second dimension of `chains`. If `pulsar_names` or `param_labels`
        are not provided, they will be generated automatically based on the
        number of pulsars and frequencies in `chains`.

        Examples
        --------
        >>> import numpy as np
        >>> from ceffyl.chain_utils import ChainProcessor
        >>> chains = np.random.randn(3, 5, 1000)  # 3 pulsars, 5 frequencies, 1000 samples
        >>> freqs = np.linspace(1e-9, 1e-6, 5)  # 5 frequencies
        >>> processor = ChainProcessor(chains=chains, freqs=freqs)
        >>> print(processor.n_psrs)  # Output: 3
        >>> print(processor.n_freqs)  # Output: 5
        >>> print(processor.n_samples)  # Output: 1000
        >>> print(processor.pulsar_names)  # Output: ['psr_1', 'psr_2', 'psr_3']
        >>> print(processor.param_labels)  # Output: ['log10_rho_1', 'log10_rho_2', ..., 'log10_rho_5']
        >>> print(processor.freqs)  # Output: [1.e-09 2.e-09 3.e-09 4.e-09 5.e-09]
        >>> print(processor.chains.shape)  # Output: (3, 5, 1000)
        
        """
        # save chains
        self.chains = chains
        self.n_psrs, self.n_freqs, self.n_samples = chains.shape
d
        if freqs.shape[0] != self.n_freqs:
            raise ValueError("Length of freqs must match the number of "
                             "frequencies in chains.")
        self.freqs = freqs

        if pulsar_names is not None:  # save pulsar names
            self.pulsar_names = pulsar_names
        elif pulsar_names is None and self.n_psrs == 1:
            # assume PTA free spectrum analysis if only one "pulsar"
            self.pulsar_names = ['freespec']
        else:
            # otherwise, assume pulsar names are in the form of 'psr_1',
            # 'psr_2', etc.
            self.pulsar_names = [f'psr_{ii+1}' for ii in range(self.n_psrs)]

        if param_labels is not None:  # save parameter labels
            self.param_labels = param_labels
        else:
            # otherwise, assume parameter labels are in the form of
            # 'log10_rho_1', 'log10_rho_2', etc.
            self.param_labels = [f'log10_rho_{ii+1}'
                                 for ii in range(self.n_freqs)]

    def bandwidth(self,
                  data: NDArray,
                  bw_func: MethodType = bw.sj_ste,
                  thin_chain: bool = False,
                  kernel_constant: float = 1.,
                  bw_kwargs: dict[str, Any] = None) -> float:
        """
        Method to calculate bandwidth for a given MCMC chain

        Parameters
        ----------
        data : NDArray
            MCMC chain to compute KDE bandwidths
        bw_func : MethodType
            function to calculate bandwidths
        thin_chain : bool
            flag to toggle thinning by autocorrelation length. Default = False
        kernel_constant : float
            A constant to transform bandwidths between one kernel to another
        bw_kwargs : dict[str, Any]
            A dict of kwargs to pass to the bandwidth function

        Returns
        -------
        bw : float
            the calculated bandwidth
        """

        if thin_chain:  # chain thinning using emcee
            thin = round(integrated_time(data)[0])
            if thin == 0:  # if thin = 0, thinning will fail
                thin = 1
        else:
            thin = 1

        # calculate bandwidth
        bw_value = bw_func(data[::thin], **bw_kwargs) * kernel_constant
        return bw_value

    def density(self,
                data: NDArray,
                bw_value: float | str,
                kernel: str = 'epa',
                thin_chain: bool = False,
                density_grid: NDArray = np.linspace(-9., -4., 10000),
                take_log: bool = True,
                reflect: bool = True,
                supress_warnings: bool = True,
                return_kde: bool = False,
                kde_kwargs: dict[str, Any] = None
                ) -> NDArray | tuple[NDArray, MethodType]:
        """
        Method to create KDE objects for an MCMC data chain

        Parameters
        ----------
        data : NDArray
            MCMC chain to construct KDEs on
        bw_value : float | str
            Bandwidth supplied to KDE. This can be a float or a string that is
            that is accepted by your KDE function
        kernel : str
            name of kernel to be used for given KDE. Default = 'epanechnikov'
        thin_chain : bool
            flag to toggle thinning of chain by autocorrelation length
        density_grid : NDArray
            np.array of log10rho grid point to calculate logpdfs of KDE
        take_log : bool
            flag to toggle returning natural log of pdf. Default = True
        reflect : bool
            boolean to include reflecting boundaries. Default = True
        supress_warnings : bool
            flag to supress warnings from taking log of 0. Default = True
        return_kde : bool
            flag to also return KDE object. Default = False
        kde_kwargs : dict[str, Any]
            keyword args for KDE methods

        Returns
        -------
        density : NDArray
            grid of logpdfs computed along log10rho grid points
        kde : MethodType, optional
            KDE object also returned if `return_kde=True`
        """
        if supress_warnings:  # supress warnings from taking log of zero
            warnings.filterwarnings('ignore')

        # if density_grid is smaller than data range, cut off data to avoid error
        data = data[data > density_grid.min()]
        data = data[data < density_grid.max()]

        if thin_chain:  # chain thinning using emcee
            thin = round(integrated_time(data)[0])
            if thin == 0:  # if thin=0, thinning will fail
                thin = 1
        else:
            thin = 1

        # initialise KDEpy.FFTKDE if chosen and fit data
        kde = FFTKDE(bw=bw_value, kernel=kernel, **kde_kwargs)

        # reflect lower boundary
        if reflect:
            lo_bound = density.min()
            processed_data = np.concatenate((data[::thin],
                                             2 * lo_bound - data[::thin]))
        else:
            processed_data = data[::thin]

        kde = kde.fit(processed_data)  # fit data

        # extend rho grid to encompass data otherwise FFTKDE fails
        drho = density_grid[1] - density_grid[0]
        density_grid_ext = np.copy(density_grid)
        while density_grid_ext.min() > processed_data.min():
            newmin = density_grid_ext[0] - drho
            density_grid_ext = np.insert(density_grid_ext, 0, newmin)

        # evaluate data on rho grid, correcting for reflection
        density = kde.evaluate(density_grid_ext)
        if reflect:
            density = density[density_grid_ext >= density_grid.min()] * 2

        if take_log:
            density = np.log(density)  # switch to take log pdf

        if return_kde:
            return (density, kde)
        else:
            return density

    # parallel processing of pulsars
    def parallelise(self,
                    psr_idx: int,
                    density_grid: NDArray = np.linspace(-9., -4., 10000),
                    bandwidth: float | MethodType | str = bw.sj_ste,
                    kernel: str = 'epa',
                    thin_chain: bool = False,
                    bw_kwargs: dict[str, Any] = None,
                    kde_kwargs: dict[str, Any] = None
                    ) -> tuple[list[NDArray], list[float]]:
        """
        A method to parallelise the computation of densities for each pulsar
        
        Parameters
        ----------
        psr_idx : int
            Index of the pulsar to compute densities for
        density_grid : NDArray
            Grid of parameter values to calculate pdfs
        bandwidth : float | MethodType | str
            Bandwidth of KDEs - may be a function to compute bandwidth, a
            precomputed float, or string associated to chosen KDE method
        kernel : str
            String name of chosen KDE kernel
        thin_chain : bool
            Toggle thinning data by autocorrelation length when fitting to kde
        bw_kwargs : dict[str, Any]
            Keyword arguments for bandwidth function
        kde_kwargs : dict[str, Any]
            Keyword arguments for KDE density function
        
        Returns
        -------
        pdfs_in : list[NDArray]
            List of probability density functions computed for each parameter
            of the pulsar
        bws_in : list[float]
            List of bandwidths computed for each parameter of the pulsar
        """
        pdfs_in, bws_in = [], []

        print(f'Computing densities for psr {psr_idx}', flush=True)

        for jj, _ in enumerate(self.param_labels):
            data = self.chains[psr_idx, jj]  # data to represent

            # calculate bandwidth if not already computed
            if isinstance(bandwidth, np.ndarray):
                bw_value = bandwidth[psr_idx][jj]
            elif not isinstance(bandwidth, (str, int, float)):
                bw_value = self.bandwidth(data, bw_func=bandwidth,
                                          thin_chain=thin_chain, **bw_kwargs)
            else:
                bw_value = bandwidth

            bws_in.append(bw_value)  # save bandwidths

            # calculate pdf along grid points and save
            pdf = self.density(data, bw, kernel=kernel,
                                density_grid=density_grid,
                                thin_chain=thin_chain, **kde_kwargs)
            pdfs_in.append(pdf)

        return pdfs_in, bws_in

    def compute_densities(self,
                          density_grid: NDArray = np.linspace(-9., -4., 10000),
                          log_infinitessimal: float = -36.,
                          outdir: str = './chain/',
                          bandwidth: float | MethodType | str = bw.sj_ste,
                          kernel: str = 'epa',
                          thin_chain: bool = False,
                          bw_kwargs: dict[str, Any] = None,
                          kde_kwargs: dict[str, Any] = None,
                          change_nans: bool = True,
                          num_threads : int = 1) -> NDArray:
        """
        A method to process an MCMC chain and compute probability densities for
        each pulsar along a grid of parameter values. Information about the
        pulsars, frequencies, and densities are saved in a PTAData object,
        which can be saved as a json file.

        Parameters
        ----------
        density_grid : NDArray
            grid of parameter values to compute pdfs along
        log_infinitessimal : float
            value to replace -inf values in pdfs. Default = -36.
        outdir : str
            directory to save the computed densities and metadata
        bandwidth : float | MethodType | str
            Bandwidth of KDEs - may be a function to compute bandwidth, a
            precomputed float, or string associated to chosen KDE method
        kernel : str
            String name of chosen KDE kernel. Default = 'epa'
        thin_chain : bool
            Toggle thinning data by autocorrelation length when fitting to kde
        bw_kwargs : dict[str, Any]
            Keyword arguments for bandwidth function
        kde_kwargs : dict[str, Any]
            Keyword arguments for KDE density function
        change_nans : bool
            If True, replace NaN values in pdfs with log_infinitessimal.
            Default = True
        num_threads : int
            Number of threads to use for parallel processing. Default = 1.
            If set to 1, no parallel processing is used.
        no_joblib : bool
            If True, do not use joblib for parallel processing. Default = False.
            If joblib is not installed, this will be set to True automatically.

        Returns
        -------
        pdfs : NDArray
            grid of logpdfs computed along log10rho grid points
        """

        # if saving densities, ensure a directory to store them before
        # significant numerical calculations!
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        
        if no_joblib:
            ansp = []
            for ii in range(self.n_psrs):
                ansp.append(self.parallelise(ii,
                                             density_grid=density_grid,
                                             bandwidth=bandwidth,
                                             kernel=kernel,
                                             thin_chain=thin_chain,
                                             bw_kwargs=bw_kwargs,
                                             kde_kwargs=kde_kwargs)
                            )
        else:
            ansp = Parallel(n_jobs=num_threads)(
                delayed(self.parallelise)(ii,
                                          density_grid=density_grid,
                                          bandwidth=bandwidth,
                                          kernel=kernel,
                                          thin_chain=thin_chain,
                                          bw_kwargs=bw_kwargs,
                                          kde_kwargs=kde_kwargs)
                    for ii in range(self.n_psrs)
                    )

        # calculating densities for each freq for each psr
        pdfs, bws = [], []
        for ii, _ in enumerate(ansp):
            pdfs.append(ansp[ii][0])
            bws.append(ansp[ii][1])
        del ansp

        pdfs = list(itertools.chain.from_iterable(pdfs))
        bw_array = list(itertools.chain.from_iterable(bws))

        # reshape array of densities
        pdfs = np.array(pdfs).reshape((self.n_psrs, self.n_freqs,
                                       len(density_grid)))
        bw_array = np.array(bw_array).reshape((self.n_psrs, self.n_freqs))

        # add infinitessimal value to avoid -inf values
        if log_infinitessimal is not None:
            infs = np.isneginf(pdfs)
            pdfs[infs] = log_infinitessimal

        if change_nans:  # if nans, convert to log infinitessimal
            print('Removing NaNs from pdfs')
            pdfs = np.nan_to_num(pdfs, nan=log_infinitessimal)

        # save metadata
        ptadata = PTAData(pulsar_names=self.pulsar_names,
                          freqs=self.freqs,
                          log_densities=pdfs,
                          density_grid=density_grid,
                          param_labels=self.param_labels,
                          tspan=None,
                          chain_processing_details={
                              'bandwidth': bw_array,
                              'kernel': kernel,
                              'thin_chain': thin_chain,
                              'bw_kwargs': bw_kwargs,
                              'kde_kwargs': kde_kwargs
                          })
        ptadata.save_as_json(os.path.join(outdir, 'ptadata.json'))

        return pdfs
