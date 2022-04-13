import numpy as np
from GFL.bw import bandwidths as bw
import acor
import la_forge.core as co
import glob

try:
    import kalepy as kale
except ImportError:
    print('kalepy cannot be found. You cannot use this if you wanted it')
    pass

from KDEpy import FFTKDE
import warnings
import os


"""
A class to create density estimators of pulsar timing array data
"""


class DE_factory:
    """
    A class to create density estimators for each rho (PSD) MCMC chain and save
    them as an array of probabilities. This is specifically designed to work on
    chains from a 'free spectrum' analysis from enterprise-pulsar
    (https://github.com/nanograv/enterprise/)
    """
    def __init__(self, coredir, la_forge=True, rho_label=None,
                 compressed_file=None, Tspan=None, N_freqs=30):
        """
        Open the compressed chain files and create density estimators

        @param coredir: directory of core objects for the GFL
                        ASSUMPTIONS - all cores have the same frequencies

        @param la_forge: use core objects from la_forge. Default: True. If
                         false, please supply a .npy/.npz file

        The following parameters will be depreciated
        @params compressed_file: location of numpy compressed file containing
                                  MCMC chains
        @param rho_label: root of labels for PSDs at each frequency
        @param Tspan: time span of the dataset
        @param N_freqs: number of frequencies of free spectrum analysis
        """

        if la_forge:
            self.la_forge = True

            corelist = glob.glob(coredir+'*.core')  # search for core

            # load core files
            cores = [co.Core(core) for core in corelist]
            self.cores = cores

            pulsar_names = [c.label for c in cores]  # get list of psr names
            self.pulsar_names = pulsar_names

            cidx = argsort(pulsar_names)  # sort psr cores alphabetically
            cores = cores[cidx]
            self.pulsar_names = pulsar_names[cidx]

            # save list of rho labels from first core
            self.rho_labels = [p for p in cores[0].params if 'rho' in p]
            self.freqs = cores[0].freqs  # save list of freqs from first core
            self.N_freqs = len(self.freqs)

        else:
            self.la_forge = False
            self.chains = np.load(compressed_file)  # load compressed chains

            # not required for calculating densities, but is important metadata
            self.freqs = np.arange(1, N_freqs+1)/Tspan

            self.N_freqs = N_freqs  # save information for later
            self.rho_labels = [rho_label+str(ii) for ii in range(N_freqs)]

            # save pulsar names from chain file
            if hasattr(self.chains, 'keys'):
                pulsar_names = list(self.chains.keys())
            else:  # if single chain from npy file, save as a single key
                pulsar_names = ['freespec']
                self.chains = dict(freespec=self.chains)

            self.pulsar_names = pulsar_names
            self.N_psrs = len(pulsar_names)

    def kernel_constants():
        """
        Here will be a dataframe to calculate the correct multiplying constants
        """
        pass

    def bandwidth(self, data, bw_func=bw.sj, thin_chain=False,
                  kernel_constant=2.214, bw_kwargs={}):
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
            thin = round(acor.acor(data)[0])

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
            thin = round(acor.acor(data)[0])

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
                data = np.concatenate((data[::thin],
                                       2 * lo_bound - data[::thin]))
                data = data[data >= lo_bound]
            else:
                data = data[::thin]

            kde = kde.fit(data)  # fit data
            density = kde.evaluate(rho_grid)

        if take_log:  # switch to take log pdf
            density = np.log(density)

        if return_kde:
            return (density, kde)

        else:
            return density

    def setup_densities(self, rho_grid=np.linspace(-15.5, 0, 1551),
                        log_infinitessimal=-20., save_density=True,
                        outdir='chain/', kde_func='FFTKDE', bandwidth=bw.sj,
                        bw_thin_chain=False, kde_thin_chain=False,
                        bw_kwargs={}, kde_kwargs={}):
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
        @param bw_kwargs: kwargs for bandwidth function
        @param kde_kwargs: kwargs for KDE density function

        @return density: array of densities
        @return kdes: array of kde objects (if chosen)
        """

        # save some properties
        self.rho_grid = rho_grid
        self.kde_func = kde_func

        if self.la_forge:
            # calculating densities for each freq for each psr
            pdfs, bws = [], []
            for c in self.cores:
                for rho in self.rho_labels:
                    data = c(rho)  # data to represent

                    # calculate bandwidth
                    if not isinstance(bandwidth, (str, int, float)):
                        bw = self.bandwidth(data, bw_func=bandwidth,
                                            thin_chain=bw_thin_chain,
                                            **bw_kwargs)
                    else:
                        bw = bandwidth

                    bws.append(bw)  # save bandwidths

                    # calculate pdf along grid points and save
                    pdfs.append(self.density(data, bw, rho_grid=rho_grid,
                                             kde_func=kde_func,
                                             thin_chain=kde_thin_chain,
                                             **kde_kwargs))

        else:  # TO BE DEPRECIATED
            # calculating densities for each freq for each psr
            pdfs, bws = [], []
            for psr in self.pulsar_names:
                print(f'Creating density array for psr {psr}')
                for rho in self.rho_labels:

                    data = self.chains[psr][rho]  # data to represent

                    # calculate bandwidth
                    if not isinstance(bandwidth, (str, int, float)):
                        bw = self.bandwidth(data, bw_func=bandwidth,
                                            thin_chain=bw_thin_chain,
                                            **bw_kwargs)
                    else:
                        bw = bandwidth

                    bws.append(bw)  # save bandwidths

                    # calculate pdf along grid points and save
                    pdfs.append(self.density(data, bw, rho_grid=rho_grid,
                                             kde_func=kde_func,
                                             thin_chain=kde_thin_chain,
                                             **kde_kwargs))

        # reshape array of densities
        pdfs = np.array(pdfs).reshape(self.N_psrs, self.N_freqs,
                                      len(rho_grid))
        self.bws = np.array(bws)

        # add infinitessimal value to avoid -inf values
        if log_infinitessimal is not None:
            infs = np.isneginf(pdfs)
            pdfs[infs] = log_infinitessimal

        self.pdfs = pdfs

        # save density and log10rho array as .npy file
        if save_density:
            self._save_densities(outdir=outdir)

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
