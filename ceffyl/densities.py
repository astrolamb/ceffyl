"""
A class to create density estimators of pulsar timing array data
"""

import numpy as np
from ceffyl.bw import bandwidths as bw
from emcee.autocorr import integrated_time
import la_forge.core as co
import glob
import time
import itertools
try:
    import kalepy as kale
except ImportError:
    print('kalepy cannot be found. You cannot use this if you wanted it')
    pass
try:
    from joblib import Parallel, delayed
    no_joblib = False
except ImportError:
    no_joblib = True
    print('joblib cannot be found. You cannot setup densities for multiple pulsars simultaneously.')
    pass
from KDEpy import FFTKDE
import warnings
import os
from natsort import natsorted


class DE_factory:
    """
    A class to create density estimators for each rho (PSD) MCMC chain and save
    them as an array of probabilities. This is specifically designed to work on
    chains from a 'free spectrum' analysis from enterprise-pulsar
    (https://github.com/nanograv/enterprise/)
    """
    def __init__(self, coredir, recursive=True, pulsar_names=[],
                 rho_labels=None):
        """
        Open the compressed chain files and create density estimators

        @param coredir: directory of core objects for the GFL
                        ASSUMPTIONS - all cores have the same frequencies
        @param recursive: a flag to note that posteriors are saved in
                          subdirectories labeled by 'psr_x'. Default:False
        @param pulsar_names: list of pulsar names
        @param rho_labels: load a subset of log10rho parameters
        """

        if recursive:  # search for cores
            corelist = natsorted(glob.glob(coredir+'/psr_**/*.core'))
        else:
            corelist = natsorted(glob.glob(coredir+'/*.core'))

        if len(corelist) == 0:
            print('No cores found!')
            return

        self.corelist = corelist

        # get list of psr names
        if len(corelist) > 1:
            self.pulsar_names = pulsar_names
            self.N_psrs = len(pulsar_names)
        elif len(corelist) == 1 and recursive:
            self.pulsar_names = pulsar_names
            self.N_psrs = 1
        else:
            self.pulsar_names = ['freespec']
            self.N_psrs = 1

        # save list of rho labels from first core
        c = co.Core(corepath=corelist[0])  # load a core
        if rho_labels is None:
            self.rho_labels = [p for p in c.params if 'rho' in p]
        else:
            self.rho_labels = rho_labels
        self.freqs = c.rn_freqs  # save list of freqs from 1st core
        self.N_freqs = len(self.freqs)

    def kernel_constants():
        """
        Here will be a dataframe to calculate the correct multiplying constants
        """
        pass

    def bandwidth(self, data, bw_func=bw.sj_ste, thin_chain=False,
                  kernel_constant=1, bw_kwargs={}):
        """
        Method to calculate bandwidth for a given MCMC chain

        @param data: MCMC chain to calculate bandwidths
        @param bw_func: function to calculate bandwidths
        @param thin_chain: flag to toggle thinning of chain by autocorrelation
                           length
        @param kernel_constant: A constant to transform bandwidths between one
                                kernel to another
        @param bw_kwargs: A dict of kwargs for the bandwidth function

        @return bw: the calculated bandwidth
        """

        # chain thinning using acor
        if thin_chain:
            thin = round(integrated_time(data)[0])

            if thin == 0:  # if acor=0, thinning will fail
                thin = 1
        else:
            thin = 1

        # calculate bandwidth
        bw = bw_func(data[::thin], **bw_kwargs) * kernel_constant

        return bw

    def density(self, data, bw, kernel='epanechnikov', kde_func='FFTKDE',
                thin_chain=False, rho_grid=np.linspace(-15.5, 0, 1551),
                take_log=True, reflect=True, supress_warnings=True,
                return_kde=False, kde_kwargs={}):
        """
        Method to create KDE objects for an MCMC data chain

        @param data: MCMC chain to calculate bandwidths
        @param bw: bandwidth of KDE. This can be a number or a string that is
                   that is accepted by your KDE function
        @param kernel: name of kernel to be used for given KDE
        @param kde_func: KDE function to be used from ['kalepy', 'FFTKDE']
        @param thin_chain: flag to toggle thinning of chain by autocorrelation
                           length
        @param KDE_kwargs: A dict of kwargs for the KDE function
        @param rho_grid: grid of log10rho values to calculate pdfs
        @param take_log: return log pdf
        @param reflect: boolean to include reflecting boundaries
        @param supress_warnings: flag to supress warnings from taking log of 0
        @param return_kde: flag to also return KDE object
        @param kde_kwargs: dict of other KDE kwargs

        @return density: return array of (log) pdfs
        @return kde: initalised KDE function if return_kde=True
        """

        if supress_warnings:  # supress warnings from taking log of zero
            warnings.filterwarnings('ignore')

        # if rho_grid is smaller than data range, cut off data to avoid error
        data = data[data > rho_grid.min()]
        data = data[data < rho_grid.max()]

        # chain thinning using acor
        if thin_chain:
            thin = round(integrated_time(data)[0])

            if thin == 0:  # if acor=0, thinning will fail
                thin = 1
        else:
            thin = 1

        # initialise kalepy if chosen and fit data
        if kde_func == 'kalepy':
            kde = kale.KDE(data[::thin], bandwidth=bw,
                           kernel=kernel, **kde_kwargs)

            if reflect:
                lo_bound = rho_grid.min()
            else:
                lo_bound = None

            density = kde.density(rho_grid, probability=True,
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
            
            # extend grid to encompass data
            drho = rho_grid[1] - rho_grid[0]
            rho_grid_ext = np.copy(rho_grid)

            while rho_grid_ext.min() > data2.min():
                newmin = rho_grid_ext[0] - drho
                rho_grid_ext = np.insert(rho_grid_ext, 0, newmin)
            
            # evaluate data on rho grid
            density = kde.evaluate(rho_grid_ext) * 2
            density = density[rho_grid_ext >= rho_grid.min()]

        if take_log:  # switch to take log pdf
            density = np.log(density)

        if return_kde:
            return (density, kde)

        else:
            return density

    def setup_densities(self, rho_grid=np.linspace(-15.5, 0, 1551),
                        log_infinitessimal=-36., save_density=True,
                        outdir='chain/', kde_func='FFTKDE', bandwidth=bw.sj_ste,
                        kernel='epanechnikov', bw_thin_chain=False,
                        kde_thin_chain=False, change_nans=True, bw_kwargs={},
                        kde_kwargs={}, bootstrap=False, Nbootstrap=None,
                        num_threads = 1):
        """
        A method to setup densitites for all chains and save them as a .npy
        file

        @param bw_thin_chain: thin data by autocorrelation length when
                              calculating bandwidth
        @param kde_thin_chain: thin data by autocorrelation length when
                               fitting to kde
        @param rho_grid: grid of log10rho values to calculate pdfs
        @param log_infinitessimal: a very small value to replace any -np.inf to
                                   allow for good sampling
        @param save_density: Flag to save rec array of densities as .npy file
        @param outdir: directory to save metadata and density array
        @param rho: path to save information about density file
        @param kde_func: KDE function to be used from ['kalepy', 'FFTKDE']
        @param bandwidth: Bandwidth of KDEs - may be a function, float, or
                          string associated to chosen KDE function
        @param change_nans: Sometimes FFTKDE will returns nans, causing issues
                            with ultranest. This changes nans to value of
                            log_infinitessimal
        @param bw_kwargs: kwargs for bandwidth function
        @param kde_kwargs: kwargs for KDE density function
        @param bootstrap: boolean to take bootstraps of samples
        @param Nbootstrap: number of samples to bootstrap if bootstrap==True
        @param num_threads: number of CPU threads to use to calculate multiple 
                            pulsars' densities simultaneously

        @return density: array of densities
        @return kdes: array of kde objects (if chosen)
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
        def doit(ii):
            pdfs_in, bws_in = [], []
            c = self.corelist[ii]
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

                # calculate bandwidth
                if isinstance(bandwidth, np.ndarray):
                    bw = bandwidth[ii][jj]
                elif not isinstance(bandwidth, (str, int, float)):
                    bw = self.bandwidth(data, bw_func=bandwidth,
                                        thin_chain=bw_thin_chain,
                                        **bw_kwargs)
                else:
                    bw = bandwidth

                bws_in.append(bw)  # save bandwidths

                # calculate pdf along grid points and save
                pdf = self.density(data, bw, kernel=kernel,
                                         rho_grid=rho_grid, kde_func=kde_func,
                                         thin_chain=kde_thin_chain,
                                         **kde_kwargs)
                pdfs_in.append(pdf)
            return pdfs_in, bws_in
        
        if no_joblib:
            ansp = []
            for ii in range(len(self.corelist)):
                ansp.append(doit(ii))
        else:
            ansp = Parallel(n_jobs=num_threads)(delayed(doit)(ii) for ii in range(len(self.corelist)))

        for ii in range(len(ansp)):
            pdfs.append(ansp[ii][0])
            bws.append(ansp[ii][1])
        del ansp
        pdfs = list(itertools.chain.from_iterable(pdfs))
        bws = list(itertools.chain.from_iterable(bws))

        # reshape array of densities
        pdfs = np.array(pdfs).reshape(self.N_psrs, self.N_freqs,
                                      len(rho_grid))
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

    def _save_densities(self, outdir, hist=False):
        """
        Method to save density array and rho grid

        @param outdir: directory to save density array, log10rho array, and
                       log10rho labels
        @param: save binedges if using histograms
        """
        if not os.path.isdir(outdir):  # check if directory exists
            os.mkdir(outdir)

        np.save(f'{outdir}/density.npy', self.pdfs)
        np.savetxt(f'{outdir}/log10rholabels.txt', self.rho_labels, fmt='%s')
        np.savetxt(f'{outdir}/pulsar_list.txt', self.pulsar_names, fmt='%s')
        np.save(f'{outdir}/freqs.npy', self.freqs)

        if hist:
            np.save(f'{outdir}/binedges.npy', self.binedges)
        else:
            np.save(f'{outdir}/bandwidths.npy', self.bws)
            np.save(f'{outdir}/log10rhogrid.npy', self.rho_grid)

        return

    def resample_ensemble(self, size=10000, freespec=False):
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
        samples = np.array([kde.resample(size=size) for kde in self.kdes])

        if not freespec:  # reshape to correct dimensions
            samples.reshape(self.N_psrs_selected, self.N_freqs, size)
            samples = samples.transpose(0, 2, 1).reshape(-1, self.N_freqs)

        return samples.T

    def histograms(self, burn=0.25, bw_thin_chain=True, hist_thin_chain=True,
                   take_log=True, log_infinitessimal=-20., save_density=True,
                   outdir='chain/', bins='fd', lowedge=-15., highedge=0.,
                   bw_kwargs={}, hist_kwargs={}):
        """
        A method to setup densitites for all chains and save them as a .npy
        file

        @param burn: number of initial samples to burn. Can be a float less
                     than 1 or an int less than number of samples
        @param bw_thin_chain: thin data by autocorrelation length when
                              calculating bandwidth
        @param hist_thin_chain: thin data by autocorrelation length when
                                fitting to histograms
        @param take_log: flag to take logpdfs
        @param log_infinitessimal: a very small value to replace any -np.inf to
                                   allow for good sampling
        @param save_density: Flag to save rec array of densities as .npy file
        @param outdir: directory to save metadata and density array
        @param bins: Calculate histogram bins - may be a function, float, or
                     string associated to np.histogram function
        @param lowedge: lowest edge of hist bins
        @param highedge: highest edge of hist bins
        @param bw_kwargs: kwargs for bandwidth function
        @param hist_kwargs: kwargs for np.histogram function

        @return density: array of densities
        @return binedges: array of histogram binedges
        """

        # calculating densities for each freq for each psr
        density, binedges, ct = [], [], 0
        for psr in self.pulsar_names:
            print(f'Creating density array for psr {ct}')
            for rho in self.rho_labels:
                print(f'Creating densities for {rho}', end='', flush=True)
                data = self.chains[psr][rho]  # data to represent

                # computing burn length
                if 0 < burn and burn < 1:
                    burn = int(burn * data.shape[0])
                elif type(burn) is int:
                    burn = burn
                else:
                    burn = 0

                data = data[burn:]

                # calculate bandwidth
                if not isinstance(bins, (str, int, float)):
                    bins = self.bandwidth(data, bw_func=bins,
                                          thin_chain=bw_thin_chain,
                                          **bw_kwargs)

                # calculate pdf along grid points and save
                pdfs, edges = np.histogram(data, bins=bins, density=True,
                                           range=(lowedge, highedge))

                if take_log:
                    pdfs = np.log(pdfs)
                    infs = np.isneginf(pdfs)
                    pdfs[infs] = log_infinitessimal

                density.append(pdfs)
                binedges.append(edges)

            ct += 1

        # reshape array of densities
        density = np.array(density, dtype=object).reshape(self.N_psrs,
                                                          self.N_freqs)
        self.pdfs = density
        self.binedges = np.array(binedges, dtype=object)

        # save density and log10rho array as .npy file
        if save_density:
            self._save_densities(outdir=outdir, hist=True)

        return density
