# imports
import numpy as np
from enterprise.signals.parameter import Uniform, Normal, LinearExp
from PTMCMCSampler.PTMCMCSampler import PTSampler as ptmcmc
from GFL import model
import os


"""
Classes to create noise signals and a parallel tempered PTMCMCSampler object
to fit spectra to a density estimation of pulsar timing array data

(\__/)
| oo |
( 00 )
"""

class signal():
    """
    A class to add signals to the GFL
    These signals can be a common process (GW) or individual to each pulsar
    (intrinsic red noise)
    """
    def __init__(self, N_freqs, selected_psrs=None, psd=model.powerlaw,
                 params=[Uniform(-18, -12)('log10_A'), Uniform(0, 7)('gamma')],
                 const_params={}, common_process=True, name='gw',
                 psd_kwargs={}):
        """
        Initialise a signal class to model intrinsic red noise or a common
        process

        //Inputs//
        @param N_freqs: Number of frequencies for this signal. Expected to be
                        equal or less than total PTA frequencies to be used

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
        self.N_freqs = N_freqs
        self.psd = psd
        self.psd_priors = params
        self.const_params = const_params
        self.psd_kwargs = psd_kwargs
        self.psr_idx = []  # to be save later in GFL class

        # save information if signal is common to all pulsars
        if common_process:
            self.CP = True
            self.selected_psrs = []
            self.param_names = [f'{p.name}_{name}' for p in params]
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
            self.param_names = [f'{q}_{name}_{p.name}' for q in
                                selected_psrs for p in params]
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
        return np.array([p.get_logpdf(x)
                         for p, x in zip(self.psd_priors, xs)]).sum()

    def get_rho(self, freqs, mapped_xs):
        """
        A method to calculate PSD of proposed values from given psd function

        //Input//
        @param freqs: Array of PTA frequencies. NOTE: function expects number
                      of frequencies to be greater than or equal to number
                      of frequencies specified for this signal (N_freqs)

        @param mapped_xs: mapped dictionary of proposed values corresponding to
                          signal params

        @return rho: array of PSDs
        """
        rho = self.psd(freqs, **mapped_xs, **self.const_params,
                       **self.psd_kwargs).T
        rho[:, self.N_freqs:] = 0  # freqs beyond N_freqs kept zero

        return rho

    def sample(self):
        """
        Method to derive a list of samples from varied parameters

        @return samples: array of samples from each parameter
        """
        return np.array([p.sample() for p in self.params])


class JumpProposal(object):
    """
    A class to propose jumps for parallel tempered swaps

    Shamelessly copied and modified from

    enterprise_extensions (https://github.com/nanograv/enterprise_extensions/)
    and
    PTMCMCSampler (https://github.com/jellis18/PTMCMCSampler/)
    """
    def __init__(self, signals):
        """
        Set up some custom jump proposals

        @params signals: list of signals for GFL
        """

        # save information as class properties
        self.params = []  # list of parameter objects
        self.param_names = []  # list of parameter names
        self.red_names = []  # list of irn parameter names
        self.gw_names = []  # list of common process parameter names

        # loop through signals and save info
        for s in signals:
            self.params.extend(s.params)
            self.param_names.extend(s.param_names)

            if s.CP:  # if signal is a CP, save names of params in this signal
                self.gw_names.extend(s.param_names)
            else:  # else, save different list for comparison purposes
                self.red_names.extend(s.param_names)

    def draw_from_prior(self, x, iter, beta):
        """
        Draw values from prior

        @param x: array of proposed parameter values
        @param iter: iteration of sampler
        @param beta: inverse temperature of chain

        @return: q: New position in parameter space
        @return: lqxy: log forward-backward jump probability
        """

        q = x.copy()
        lqxy = 0

        # randomly choose parameter
        pidx = np.random.randint(0, len(x))

        # sample this parameter
        p = self.params[pidx]
        q[pidx] = p.sample()

        # forward-backward jump probability
        lqxy = p.get_logpdf(x[pidx] - q[pidx])

        return q, float(lqxy)

    def draw_from_red_prior(self, x, iter, beta):
        """
        Prior draw from red noise

        @param x: array of proposed parameter values
        @param iter: iteration of sampler
        @param beta: inverse temperature of chain

        @return: q: New position in parameter space
        @return: lqxy: log forward-backward jump probability
        """

        q = x.copy()
        lqxy = 0

        # randomly choose parameter
        p_name = np.random.choice(self.red_names)
        pidx = self.param_names.index(p_name)
        p = self.params[pidx]

        # sample this parameter
        q[pidx] = p.sample()

        # forward-backward jump probability
        lqxy = p.get_logpdf(x[pidx] - q[pidx])

        return q, float(lqxy)

    def draw_from_gwb_log_uniform_dist(self, x, iter, beta):
        """
        Prior draw from log uniform GWB distribution

        @param x: array of proposed parameter values
        @param iter: iteration of sampler
        @param beta: inverse temperature of chain

        @return: q: New position in parameter space
        @return: lqxy: log forward-backward jump probability
        """

        q = x.copy()
        lqxy = 0

        # randomly choose parameter
        p_name = np.random.choice(self.gw_names)
        pidx = self.param_names.index(p_name)
        p = self.params[pidx]

        # sample this parameter
        q[pidx] = np.random.uniform(p.prior._defaults['pmin'],
                                    p.prior._defaults['pmax'])

        # forward-backward jump probability
        lqxy = p.get_logpdf(x[pidx] - q[pidx])

        return q, float(lqxy)


class GFL():
    """
    The Generalised Factorised Likelihood (GFL) Method

    A class to fit signals to single pulsar free spectra via PTMCMC to derive
    the signals' spectral characteristics
    """
    def __init__(self, densitydir, pulsar_list=None, hist=False):
        """
        Initialise the GFL

        @params density_file: location of numpy arrays containing densities,
                              corresponding log10rho grids and labels, chain
                              names, and frequencies
        @params pulsar_list: list of pulsars to search over
        @param hist: Flag to state that you're using histograms instead of
                     KDEs
        """

        # saving properties
        self.freqs = np.load(f'{densitydir}/freqs.npy')
        self.N_freqs = self.freqs.size
        self.reshaped_freqs = self.freqs.reshape((1, self.N_freqs)).T
        self.Tspan = self.freqs[0]
        self.rho_labels = np.loadtxt(f'{densitydir}/log10rholabels.txt',
                                     dtype=np.unicode_, ndmin=1)
        if hist:
            self.binedges = np.load(f'{densitydir}/binedges.npy',
                                    allow_pickle=True)
        else:
            self.rho_grid = np.load(f'{densitydir}/log10rhogrid.npy')

        # selected pulsars
        if pulsar_list is None:
            self.pulsar_list = list(np.loadtxt(f'{densitydir}/pulsar_list.txt',
                                               dtype=np.unicode_, ndmin=1))
        else:
            self.pulsar_list = pulsar_list  # is list if none

        # find index of sublist
        file_psrs = list(np.loadtxt(f'{densitydir}/pulsar_list.txt',
                                    dtype=np.unicode_,
                                    ndmin=1))
        selected_psrs = [file_psrs.index(p) for p in self.pulsar_list]
        self.N_psrs = len(self.pulsar_list)

        # load densities from npy binary file for given psrs, freqs
        density_file = f'{densitydir}/density.npy'
        density = np.load(density_file, allow_pickle=True)[selected_psrs]

        self.density = density

    def setup_sampler(self, signals, outdir, logL, logp, resume=True,
                      jump=True, groups=None, loglkwargs={}, logpkwargs={},
                      ptmcmc_kwargs={}):
        """
        Method to setup sampler

        //Inputs//
        @params signals: A list of signals to be searched over
        @params outdir: Path to directory to save MCMC chain
        @params logL: Log likelihood function for the MCMC
        @params logp: Log prior function for the MCMC
        @params resume: Flag to toggle option to resume MCMC run from a
                        previous run
        @params jump: Flag to use jump proposals in parallel tempering
        @params groups: indices for which to perform adaptive jumps
        @param loglkwargs: additional kwargs for log likelihood
        @param logpkwargs: additional kwargs for log prior
        @param ptmcmc_kwargs: additional kwargs for PTMCMCSampler

        @return sampler: initialised PTMCMC sampler
        """

        # check if pulsars in signals are in total pulsar array
        for s in signals:
            if not np.isin(s.selected_psrs, self.pulsar_list).all():
                print('Mismatch between density array pulsars and the pulsars'
                      + ' you selected')
                return
            else:  # save idx of (subset of) psrs within larger list
                if s.CP:
                    s.psr_idx = np.arange(self.N_psrs)
                else:
                    s.psr_idx = np.array([self.pulsar_list.index(p)
                                          for p in s.selected_psrs])

        # save array of signals
        self.signals = signals

        # save complete array of parameters
        self.param_names = list(np.hstack([s.param_names for s in signals]))
        ndim = len(self.param_names)

        # setup empty 2d grid to vectorize product of pdfs
        self._I, self._J = np.ogrid[:self.N_psrs, :self.N_freqs]

        # initial jump covariance matrix
        if os.path.exists(outdir+'/cov.npy'):
            cov = np.load(outdir+'/cov.npy')
        else:
            cov = np.diag(np.ones(ndim) * 0.1**2)

        # group params for PT swaps
        if groups is None:
            groups = [list(np.arange(0, ndim))]

            # make a group for each signal, with all non-global parameters
            ct = 0
            for s in signals:
                groups.append(list(np.arange(ct, ct+s.length)))

                if s.CP:  # visit GW signals x5 more often
                    [groups.append(list(np.arange(ct, ct+s.length)))
                     for ii in range(5)]

                else:  # group individual pulsars
                    ct2 = ct
                    for jj in range(s.N_psrs):
                        groups.append(list(np.arange(ct2,
                                                     ct2+len(s.psd_priors))))
                        ct2 += len(s.psd_priors)

                ct += s.length

        # sampler
        sampler = ptmcmc(ndim, logL, logp, cov, outDir=outdir, resume=resume,
                         loglkwargs=loglkwargs, logpkwargs=logpkwargs,
                         groups=groups, **ptmcmc_kwargs)

        # save parameter names
        np.savetxt(outdir+'/pars.txt', self.param_names, fmt='%s')

        # PT swap jump proposals
        if jump:
            jp = JumpProposal(signals)
            sampler.jp = jp

            # always add draw from prior
            sampler.addProposalToCycle(jp.draw_from_prior, 5)

            # flags to automatically add prior draws given certain signals
            red_noise, gw_signal = False, False
            for s in signals:
                if s.CP:
                    gw_signal = True
                else:
                    red_noise = True

            # Red noise prior draw
            if red_noise:
                print('Adding red noise prior draws...\n')
                sampler.addProposalToCycle(jp.draw_from_red_prior, 10)

            # GWB uniform distribution draw
            if gw_signal:
                print('Adding GWB uniform distribution draws...\n')
                sampler.addProposalToCycle(jp.draw_from_gwb_log_uniform_dist,
                                           10)

        return sampler

    def initial_samples(self):
        """
        A method to return an array of initial random samples for PTMCMC

        @return x0: array of initial samples
        """
        x0 = np.hstack([s.sample() for s in self.signals])
        return x0

    def ln_prior(self, xs):
        """
        vectorised log prior function for PTMCMC to calculate logpdf of
        proposed values given their parameter distribution

        @param xs: proposed value array

        @return logpdf: total logpdf of proposed values given signal parameters
        """
        ct = 0  # parameter counter
        logpdf = 0  # total logpdf
        for s in self.signals:  # iterate through signals
            # reshape array to vectorise to size (N_kwargs, N_sig_psrs)
            x = xs[ct:ct+s.length]
            mapped_x = np.array([x[ii::len(s.psd_priors)]
                                for ii in range(len(s.psd_priors))])
            logpdf += s.get_logpdf(mapped_x)
            ct += s.length

        return logpdf

    def ln_likelihood(self, xs):
        """
        vectorised log likelihood function for PTMCMC to calculate logpdf of
        proposed values given KDE density array

        @param xs: proposed value array

        @return logpdf: total logpdf of proposed values given KDE density array
        """
        ct = 0  # parameter counter
        rho = np.zeros((self.N_psrs, self.N_freqs))  # initalise empty array
        for s in self.signals:  # iterate through signals
            # reshape array to vectorise to size (N_kwargs, N_sig_psrs)
            x = xs[ct:ct+s.length]
            mapped_x = {s_i.name: x[ii::len(s.psd_priors)]
                        for ii, s_i in enumerate(s.psd_priors)}
            rho[s.psr_idx] += s.get_rho(self.reshaped_freqs, mapped_x)
            ct += s.length

        logrho = 0.5*np.log10(rho)  # calculate log10 root PSD

        # search for location of logrho values within grid
        idx = np.searchsorted(self.rho_grid, logrho) - 1

        # access those logpdf values from density array and sum it
        logpdf = self.density[self._I, self._J, idx]

        return np.sum(logpdf)

    def hist_ln_likelihood(self, xs):
        """
        log likelihood function for PTMCMC to calculate logpdf of
        proposed values given histogram density arrays. This isn't optimised
        for speed. It is best for PTA freespec only

        @param xs: proposed value array

        @return logpdf: total logpdf of proposed values given KDE density array
        """
        ct = 0  # parameter counter
        rho = np.zeros((self.N_psrs, self.N_freqs))  # initalise empty array
        for s in self.signals:  # iterate through signals
            # reshape array to vectorise to size (N_kwargs, N_sig_psrs)
            x = xs[ct:ct+s.length]
            mapped_x = {s_i.name: x[ii::len(s.psd_priors)]
                        for ii, s_i in enumerate(s.psd_priors)}
            rho[s.psr_idx] += s.get_rho(self.reshaped_freqs, mapped_x)
            ct += s.length

        logrho = 0.5*np.log10(rho)  # calculate log10 root PSD

        # search for location of logrho values within grid and logpdf
        logpdf = 0
        for ii in range(self.N_psrs):
            for jj in range(self.N_freqs):
                idx = np.searchsorted(self.binedges[ii*self.N_freqs+jj],
                                      logrho[ii][jj]) - 1
                logpdf += self.density[ii][jj][idx]

        return logpdf
