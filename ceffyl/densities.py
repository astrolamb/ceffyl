"""
A class to create density estimators of pulsar timing array data
"""

import numpy as np
from ceffyl.bw import bandwidths as bw
from emcee.autocorr import integrated_time
import la_forge.core as co
import glob
import time
from KDEpy import FFTKDE
import warnings
import os
from natsort import natsorted
from numpy.typing import NDArray
from types import MethodType
from typing import Any

try:
    import kalepy as kale
except ImportError:
    print('kalepy cannot be found. You cannot use this if you wanted it')
    pass


class DE_factory:
    """
    A class to create density estimations for each log10rho (PSD) MCMC chain and
    save them as an array of probabilities. This is specifically designed to
    work on chains from an MCMC 'free spectrum' analysis from enterprise-pulsar
    (https://github.com/nanograv/enterprise/)
    """
    def __init__(self,
                 coredir: str,
                 recursive: bool = True,
                 pulsar_names: list[str] = [],
                 rho_labels: list[str] = None):
        """
        Open the compressed chain files and create density estimators

        Parameters
        ----------
        coredir : str
            directory of la-forge core objects for Ceffyl
            ASSUMPTIONS - all cores have the same frequencies
        recursive : bool
            a flag to note that posteriors are saved in subdirectories labeled
            by 'psr_x'. Default: False
        pulsar_names : list[str]
            list of pulsar names to save
        rho_labels : list[str]
            load a subset of log10rho parameters

        Raise
        -----
        KeyError
            if no la-forge corefiles are found
        """

        # when WGL runs single pulsar analyses, he saves them in an recursive
        #   directory system, i.e. a directory system that looks like this:
        #
        # - single_pulsar_analyses -> psr_0 -> chain.core
        #                          -> psr_1 -> chain.core
        #                          -> psr_2 -> chain.core
        #                          ...
        # To process all of these chains together, use the `recursive` flag
        #
        # Else, for a PTA free spectrum analysis (i.e. one free spectrum for
        #   the entire PTA data set), there is only one chain to process, hence
        #   `recursive` mode isn't required

        # search for cores in given coredir and save their locations
        if recursive:
            corelist = natsorted(glob.glob(coredir+'/psr_**/*.core'))
        else:
            corelist = natsorted(glob.glob(coredir+'/*.core'))
        if len(corelist) == 0:
            raise KeyError('No cores found!')
        self.corelist = corelist

        # save list of pulsar names
        if len(corelist) > 1:
            self.pulsar_names = pulsar_names
            self.N_psrs = len(pulsar_names)
        elif len(corelist) == 1 and recursive:
            self.pulsar_names = pulsar_names
            self.N_psrs = 1
        else:  # for a free spec analysis, assign 'pulsar name' as 'freespec'
            self.pulsar_names = ['freespec']
            self.N_psrs = 1

        # save list of rho labels from first core
        # we are assuming that freespec frequencies are the same across pulsars
        c = co.Core(corepath=corelist[0])  # load a core
        if rho_labels is None:
            self.rho_labels = [p for p in c.params if 'rho' in p]
        else:
            self.rho_labels = rho_labels
        self.freqs = c.rn_freqs  # save list of freqs from 1st core
        self.N_freqs = len(self.freqs)


    def bandwidth(self,
                  data: NDArray,
                  bw_func: MethodType = bw.sj,
                  thin_chain: bool = False,
                  kernel_constant: float = 1.,
                  bw_kwargs: dict[str, Any] = {}) -> float:
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
        bw = bw_func(data[::thin], **bw_kwargs) * kernel_constant
        return bw

    def density(self, 
                data: NDArray,
                bw: float | str,
                kernel: str = 'epanechnikov',
                kde_func: str = 'FFTKDE',
                thin_chain: bool = False,
                rho_grid: NDArray = np.linspace(-9., -4., 10000),
                take_log: bool = True,
                reflect: bool = True,
                supress_warnings: bool = True,
                return_kde: bool = False,
                kde_kwargs: dict[str, Any] = {}
                ) -> NDArray | (NDArray, MethodType):
        """
        Method to create KDE objects for an MCMC data chain

        Parameters
        ----------
        data : NDArray
            MCMC chain to construct KDEs on
        bw : float | str
            Bandwidth supplied to KDE. This can be a float or a string that is
            that is accepted by your KDE function
        kernel : str
            name of kernel to be used for given KDE. Default = 'epanechnikov'
        kde_func : str
            KDE method to be used from ['kalepy', 'FFTKDE']
        thin_chain : bool
            flag to toggle thinning of chain by autocorrelation length
        rho_grid : NDArray
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

        # if rho_grid is smaller than data range, cut off data to avoid error
        data = data[data > rho_grid.min()]
        data = data[data < rho_grid.max()]

        if thin_chain:  # chain thinning using emcee
            thin = round(integrated_time(data)[0])
            if thin == 0:  # if thin=0, thinning will fail
                thin = 1
        else:
            thin = 1

        # initialise kalepy if chosen and fit data
        if kde_func == 'kalepy':
            kde = kale.KDE(data[::thin], bandwidth=bw, kernel=kernel,
                           **kde_kwargs)
            lo_bound = rho_grid.min() if reflect else None  # set up boundary
            density = kde.density(rho_grid,                 # reflection
                                  probability=True,
                                  reflect=[lo_bound, None])[1]

        # initialise KDEpy.FFTKDE if chosen and fit data
        elif kde_func == 'FFTKDE':
            if kernel == 'epanechnikov':  # change name for FFTKDE
                kernel = 'epa'
            kde = FFTKDE(bw=bw, kernel=kernel, **kde_kwargs)

            # reflect lower boundary
            if reflect:
                lo_bound = rho_grid.min()
                data2 = np.concatenate((data[::thin],
                                       2 * lo_bound - data[::thin]))
            else:
                data2 = data[::thin]

            kde = kde.fit(data2)  # fit data
            
            # extend rho grid to encompass data otherwise FFTKDE fails
            drho = rho_grid[1] - rho_grid[0]
            rho_grid_ext = np.copy(rho_grid)
            while rho_grid_ext.min() > data2.min():
                newmin = rho_grid_ext[0] - drho
                rho_grid_ext = np.insert(rho_grid_ext, 0, newmin)
            
            # evaluate data on rho grid, correcting for reflection
            density = kde.evaluate(rho_grid_ext)
            if reflect:
                density = density[rho_grid_ext >= rho_grid.min()] * 2

        if take_log: density = np.log(density)  # switch to take log pdf

        if return_kde:
            return (density, kde)
        else:
            return density

    def setup_densities(self,
                        rho_grid: NDArray = np.linspace(-15.5, 0, 1551),
                        log_infinitessimal: float = -36.,
                        save_density: bool = True,
                        outdir: str = 'chain/',
                        kde_func: str = 'FFTKDE',
                        bandwidth: float | MethodType | str = bw.sj,
                        kernel: str = 'epanechnikov',
                        bw_thin_chain: bool = False,
                        kde_thin_chain: bool = False,
                        change_nans: bool = True,
                        bw_kwargs: dict[str, Any] = {},
                        kde_kwargs: dict[str, Any] = {},
                        bootstrap: bool = False,
                        Nbootstrap: int = None) -> NDArray:
        """
        A method to setup densitites for all chains and save them as a .npy
        file

        Parameters
        ----------
        rho_grid : NDArray
            grid of log10rho values to calculate pdfs
        log_infinitessimal : float
            a very small value to replace any -np.inf to allow for good sampling
        save_density : bool
            flag to save rec array of densities as .npy file
        outdir : str
            directory to save metadata and density array
        kde_func : str
            KDE function to be used from ['FFTKDE', 'kalepy']
        bandwidth : MethodType | float | str
            Bandwidth of KDEs - may be a function to compute bandwidth, a
            precomputed float, or string associated to chosen KDE method
        kernel : str
            string name of chosen KDE kernel
        bw_thin_chain : bool 
            toggle thinning data by autocorrelation length when computing
            bandwidth. Default = False
        kde_thin_chain : bool
            toggle thinning data by autocorrelation length when fitting to kde
        change_nans : bool
            Sometimes FFTKDE will returns nans, causing issues with ultranest.
            Flag changes nans to value of log_infinitessimal. Default = True
        bw_kwargs : dict[str, Any]
            keyword arguments for bandwidth function
        kde_kwargs : dict[str, Any]
            keyword arguments for KDE density function
        bootstrap : bool
            flag to bootstrap samples
        Nbootstrap : int
            number of samples to bootstrap if `bootstrap=True`

        Returns
        -------
        pdfs : NDArray
            grid of logpdfs computed along log10rho grid points
        """

        # if saving densities, ensure a directory to store them before
        # significant numerical calculations!
        if save_density:
            if not os.path.exists(outdir):
                os.makedirs(outdir)

        # save some properties
        self.rho_grid = rho_grid
        self.kde_func = kde_func

        # calculating densities for each freq for each psr
        pdfs, bws = [], []
        for ii, c in enumerate(self.corelist):
            print(f'Computing densities for psr {ii}', flush=True)
            core = co.Core(corepath=c)

            for jj, rho in enumerate(self.rho_labels):
                data = core(rho)  # data to represent

                if bootstrap:  # bootstrap data
                    if Nbootstrap is None:
                        Nbootstrap = data.shape[0]

                    bootmask = np.random.randint(0, high=Nbootstrap,
                                                 size=Nbootstrap)
                    data = data[bootmask]

                # calculate bandwidth if not already computed
                if isinstance(bandwidth, np.ndarray):
                    bw = bandwidth[ii][jj]
                elif not isinstance(bandwidth, (str, int, float)):
                    bw = self.bandwidth(data, bw_func=bandwidth,
                                        thin_chain=bw_thin_chain,
                                        **bw_kwargs)
                else:
                    bw = bandwidth
                bws.append(bw)  # save bandwidths

                # calculate pdf along grid points and save
                pdfs.append(self.density(data, bw, kernel=kernel,
                                         rho_grid=rho_grid, kde_func=kde_func,
                                         thin_chain=kde_thin_chain,
                                         **kde_kwargs))

        # reshape array of densities
        pdfs = np.array(pdfs).reshape(self.N_psrs, self.N_freqs, len(rho_grid))
        self.bws = np.array(bws)

        # add infinitessimal value to avoid -inf values
        if log_infinitessimal is not None:
            infs = np.isneginf(pdfs)
            pdfs[infs] = log_infinitessimal

        if change_nans:  # if nans, convert to log infinitessimal
            print('removing nans')
            pdfs = np.nan_to_num(pdfs, nan=log_infinitessimal)
        self.pdfs = pdfs

        # save density and log10rho array as .npy file
        if save_density:
            self._save_densities(outdir=outdir)

        # save log with information
        with open(outdir+'/log.txt', 'w') as f:
            f.write(f'Date created:{time.localtime()}\n')
            f.write(f'la-forge cores: {self.corelist}\n')
            f.write(f'KDE function: {self.kde_func}\n')
            f.write(f'KDE kernel choice: {kernel}\n')
            f.write(f'Bandwidth choice: {bandwidth}\n')
            f.write(f'Bandwidth values: {self.bws}\n')
            f.write(f'log10rho grid: {self.rho_grid}\n')
            f.write(f'Thin chain for KDE? {kde_thin_chain}\n')
            f.write(f'Thin chain for bandwidth? {bw_thin_chain}\n')
            f.write(f'Bootstrap? {bootstrap}\n')
            f.write(f'Number of bootstrapped samples: {Nbootstrap}\n')

        return pdfs

    def _save_densities(self, outdir: str):
        """
        Method to save density array and rho grid

        @param outdir: directory to save density array, log10rho array, and
                       log10rho labels
        @param: save binedges if using histograms
        """
        if not os.path.isdir(outdir):  # check if directory exists
            os.makedirs(outdir)

        np.save(f'{outdir}/density.npy', self.pdfs)
        np.savetxt(f'{outdir}/log10rholabels.txt', self.rho_labels, fmt='%s')
        np.savetxt(f'{outdir}/pulsar_list.txt', self.pulsar_names, fmt='%s')
        np.save(f'{outdir}/freqs.npy', self.freqs)
        np.save(f'{outdir}/bandwidths.npy', self.bws)
        np.save(f'{outdir}/log10rhogrid.npy', self.rho_grid)

        return

    def resample_ensemble(self,
                          kdes: MethodType,
                          size: int = 100000,
                          freespec: bool = False) -> NDArray:
        """
        A method to resample from all KDEs
        NOTE: This only works for kalepy

        @param size: number of samples
        @param freespec: flag to check if this is only for a single psr or
                         free spectrum search. Select 'True' is so. This
                         ensures shape of returned array is correct

        @return samples: array of samples
        """
        # resample
        samples = np.array([kde.resample(size=size) for kde in kdes])

        if not freespec:  # reshape to correct dimensions
            samples.reshape(self.N_psrs_selected, self.N_freqs, size)
            samples = samples.transpose(0, 2, 1).reshape(-1, self.N_freqs)

        return samples.T
