################################################################################
# ceffylGP
# A script to fit trained GPs to PTA free spectrum
# Author: William G. Lamb 2023
################################################################################

# imports
import numpy as np
import h5py
import pickle
from enterprise.signals import parameter
from enterprise_extensions import sampler
from enterprise_extensions.sampler import extend_emp_dists
from ceffyl import Ceffyl, models
from PTMCMCSampler.PTMCMCSampler import PTSampler as ptmcmc
from holodeck.gps import gp_utils
from scipy.special import logsumexp
from enterprise.signals import gp_priors
################################################################################
class JumpProposal(object):
    """
    A class to propose jumps for parallel tempered swaps
    Shamelessly copied and modified from
    enterprise_extensions (https://github.com/nanograv/enterprise_extensions/)
    and
    PTMCMCSampler (https://github.com/jellis18/PTMCMCSampler/)
    """
    def __init__(self, ceffylGP, empirical_distr=None):
        """
        Set up some custom jump proposals

        @params ceffylGP - import an initialised ceffylGP class
        @params empirical_distr: a list of log10_rho empirical distributions
        """

        # save information as class properties
        self.params = ceffylGP.params  # list of parameter objects
        self.param_names = ceffylGP.param_names  # list of parameter names
        self.hyperparams = ceffylGP.hypervar  # list of gp param names
        self.hypernames = [h.name for h in ceffylGP.hypervar]
        self.empirical_distr = empirical_distr

        # parameter indices map
        self.pimap = {}
        for ct, p in enumerate(self.param_names):
            self.pimap[p] = ct
            
        # parameter map
        self.pmap, ct = {}, 0
        for p in self.params:
            size = p.size or 1
            self.pmap[str(p)] = slice(ct, ct+size)
            ct += size
            
        # only save the empirical distributions for
        # parameters that are in the model
        if empirical_distr is not None:
            mask = []
            for idx, d in enumerate(empirical_distr):
                if d.ndim == 1:
                    if d.param_name in self.param_names:
                        mask.append(idx)
                else:
                    if (d.param_name[0] in self.param_names 
                        and d.param_name[1] in self.param_names):
                        mask.append(idx)
            if len(mask) >= 1:
                self.empirical_distr = [empirical_distr[m] for m in mask]
                # some empirical distribution do not cover the entire parameter
                # space... extend empirical_distr here
                print('Extending empirical distributions to priors...\n')
                self.empirical_distr = extend_emp_dists(ceffylGP,
                                                        self.empirical_distr,
                                                        npoints=100_000)
            else:
                self.empirical_distr = None

    def draw_from_prior(self, x, iter, beta):
        """
        draw samples from a random parameter prior
        The function signature is specific to PTMCMCSampler.
        """
        q = x.copy()  # copy proposed value array
        lqxy = 0  # set jump probability to zero

        param = np.random.choice(self.params)  # randomly choose parameter

        if param.size:  # if vector parameter jump in random component
            idx2 = np.random.randint(0, param.size)
            q[self.pmap[str(param)]][idx2] = param.sample()[idx2]

        else:  # scalar parameter
            q[self.pmap[str(param)]] = param.sample()

        # forward-backward jump probability
        lqxy = (param.get_logpdf(x[self.pmap[str(param)]]) -
                param.get_logpdf(q[self.pmap[str(param)]]))

        return q, float(lqxy)

    """
    def draw_from_gw_rho_prior(self, x, iter, beta):
        #
        draw samples from a free spectrum prior
        The function signature is specific to PTMCMCSampler.
        #
        q = x.copy()  # copy proposed value array
        lqxy = 0  # set jump probability to zero
        param = self.log10_rho  # free spectrum parameter

        if param.size:  # if vector parameter jump in random component
            idx2 = np.random.randint(0, param.size)
            q[self.pmap[str(param)]][idx2] = param.sample()[idx2]

        else:  # scalar parameter
            q[self.pmap[str(param)]] = param.sample()

        # forward-backward jump probability
        lqxy = (param.get_logpdf(x[self.pmap[str(param)]]) -
                param.get_logpdf(q[self.pmap[str(param)]]))

        return q, float(lqxy)
    """
    
    def draw_from_env_prior(self, x, iter, beta):
        """
        draw samples from a GP environment prior
        The function signature is specific to PTMCMCSampler.
        """
        q = x.copy()  # copy proposed value array
        lqxy = 0  # set jump probability to zero
        param = np.random.choice(self.hyperparams)  # randomly choose parameter

        if param.size:   # if vector parameter jump in random component
            idx2 = np.random.randint(0, param.size)
            q[self.pmap[str(param)]][idx2] = param.sample()[idx2]

        else:  # scalar parameter
            q[self.pmap[str(param)]] = param.sample()

        # forward-backward jump probability
        lqxy = (param.get_logpdf(x[self.pmap[str(param)]]) -
                param.get_logpdf(q[self.pmap[str(param)]]))

        return q, float(lqxy)
    
    def draw_from_hard_prior(self, x, iter, beta):
        """
        draw samples from a GP environment prior with label 'hard'
        The function signature is specific to PTMCMCSampler.
        """
        q = x.copy()  # copy proposed value array
        lqxy = 0  # set jump probability to zero

        # draw parameter from signal model
        pname = np.random.choice([pnm for pnm in self.hypernames
                                  if 'hard' in pnm])
        idx = self.hypernames.index(pname)
        param = self.hyperparams[idx]

        if param.size:   # if vector parameter jump in random component
            idx2 = np.random.randint(0, param.size)
            q[self.pmap[str(param)]][idx2] = param.sample()[idx2]

        else:  # scalar parameter
            q[self.pmap[str(param)]] = param.sample()

        # forward-backward jump probability
        lqxy = (param.get_logpdf(x[self.pmap[str(param)]]) -
                param.get_logpdf(q[self.pmap[str(param)]]))

        return q, float(lqxy)
    
    def draw_from_gsmf_prior(self, x, iter, beta):
        """
        draw samples from a GP environment prior with label 'gsmf'
        The function signature is specific to PTMCMCSampler.
        """
        q = x.copy()  # copy proposed value array
        lqxy = 0  # set jump probability to zero

        # draw parameter from signal model
        pname = np.random.choice([pnm for pnm in self.hypernames
                                  if 'gsmf' in pnm])
        idx = self.hypernames.index(pname)
        param = self.hyperparams[idx]

        if param.size:   # if vector parameter jump in random component
            idx2 = np.random.randint(0, param.size)
            q[self.pmap[str(param)]][idx2] = param.sample()[idx2]

        else:  # scalar parameter
            q[self.pmap[str(param)]] = param.sample()

        # forward-backward jump probability
        lqxy = (param.get_logpdf(x[self.pmap[str(param)]]) -
                param.get_logpdf(q[self.pmap[str(param)]]))

        return q, float(lqxy)
    
    def draw_from_mmb_prior(self, x, iter, beta):
        """
        draw samples from a GP environment prior with label 'mmb'
        The function signature is specific to PTMCMCSampler.
        """
        q = x.copy()  # copy proposed value array
        lqxy = 0  # set jump probability to zero

        # draw parameter from signal model
        pname = np.random.choice([pnm for pnm in self.hypernames
                                  if 'mmb' in pnm])
        idx = self.hypernames.index(pname)
        param = self.hyperparams[idx]

        if param.size:   # if vector parameter jump in random component
            idx2 = np.random.randint(0, param.size)
            q[self.pmap[str(param)]][idx2] = param.sample()[idx2]

        else:  # scalar parameter
            q[self.pmap[str(param)]] = param.sample()

        # forward-backward jump probability
        lqxy = (param.get_logpdf(x[self.pmap[str(param)]]) -
                param.get_logpdf(q[self.pmap[str(param)]]))

        return q, float(lqxy)
    
    def draw_from_gpf_prior(self, x, iter, beta):
        """
        draw samples from a GP environment prior with label 'gpf'
        The function signature is specific to PTMCMCSampler.
        """
        q = x.copy()  # copy proposed value array
        lqxy = 0  # set jump probability to zero

        # draw parameter from signal model
        pname = np.random.choice([pnm for pnm in self.hypernames
                                  if 'gpf' in pnm])
        idx = self.hypernames.index(pname)
        param = self.hyperparams[idx]

        if param.size:   # if vector parameter jump in random component
            idx2 = np.random.randint(0, param.size)
            q[self.pmap[str(param)]][idx2] = param.sample()[idx2]

        else:  # scalar parameter
            q[self.pmap[str(param)]] = param.sample()

        # forward-backward jump probability
        lqxy = (param.get_logpdf(x[self.pmap[str(param)]]) -
                param.get_logpdf(q[self.pmap[str(param)]]))

        return q, float(lqxy)
    
    def draw_from_gmt_prior(self, x, iter, beta):
        """
        draw samples from a GP environment prior with label 'hard'
        The function signature is specific to PTMCMCSampler.
        """
        q = x.copy()  # copy proposed value array
        lqxy = 0  # set jump probability to zero

        # draw parameter from signal model
        pname = np.random.choice([pnm for pnm in self.hypernames
                                  if 'gmt' in pnm])
        idx = self.hypernames.index(pname)
        param = self.hyperparams[idx]

        if param.size:   # if vector parameter jump in random component
            idx2 = np.random.randint(0, param.size)
            q[self.pmap[str(param)]][idx2] = param.sample()[idx2]

        else:  # scalar parameter
            q[self.pmap[str(param)]] = param.sample()

        # forward-backward jump probability
        lqxy = (param.get_logpdf(x[self.pmap[str(param)]]) -
                param.get_logpdf(q[self.pmap[str(param)]]))

        return q, float(lqxy)

################################################################################
class ceffylGP():
    """
    A class to fit GP to PTA free spectrum via ceffyl
    """
    def __init__(self, datadir, hyperparams, gp, gp_george,
                 Nfreqs=None, freq_idxs=None, log10_rho_priors=[-10, -6]):
        """
        Initialise the class

        @param datadir: Path to ceffyl free spectrum representation directory
        @param Nfreqs: Number of frequencies to fit
        @param hyperparams: A list of environment parameters to fit into GP.
                            NOTE: this must be in the correct shape and order
                            for the GP. Use parameter.Constant for setting
                            constant values.
        @param gp: a list of GP objects
        @param gp_george: a list of george objects
        @param log10_rho_priors: a list of the prior boundaries for log10_rho
        """
        # save data
        self.datadir = datadir
        self.Nfreqs = Nfreqs
        self.hyperparams = hyperparams
        self.gp = gp
        self.gp_george = gp_george
        
        # create ceffyl object w/ Nfreq free spec
        ceffyl_pta = Ceffyl.ceffyl(datadir)
        
        # log10rho parameter to initialise ceffyl object
        log10_rho = parameter.Uniform(*log10_rho_priors, 
                                      size=Nfreqs)('log10_rho')
        gw = Ceffyl.signal(psd=models.free_spectrum, N_freqs=Nfreqs,
                          params=[log10_rho], freq_idxs=freq_idxs)
        ceffyl_pta = ceffyl_pta.add_signals([gw])
        self.ceffyl_pta = ceffyl_pta  # save ceffyl object
        
        self.Tspan = 1/ceffyl_pta.freqs[0]  # PTA frequencies being searched
        self.freqs = ceffyl_pta.freqs[:Nfreqs]  # save frequencies

        # save rho grid
        #rho_mask = (self.ceffyl_pta.rho_grid > log10_rho_priors[0] and 
        #            self.ceffyl_pta.rho_grid < log10_rho_priors[1])
        rho_grid = self.ceffyl_pta.rho_grid
        self.rho_grid = np.repeat([rho_grid], repeats=Nfreqs,
                                  axis=0) # save freespec probability grid

        # saving locations of constant hyperparams
        const_idx = np.where(np.array([hasattr(h, 'sample')
                                       for h in hyperparams])==False)[0]
        const_values = np.array([h.value for h
                                 in np.array(hyperparams)[const_idx]])
        self.const_idx, self.const_values = const_idx, const_values
        
        # locations of variable hyperparams
        self.hypervar_idx = np.where(np.array([hasattr(h, 'sample')
                                               for h in hyperparams])==True)[0]
        self.hypervar = np.array(hyperparams)[self.hypervar_idx]

        # saving params that can be sampled (i.e. no constant values)
        self.params = list(self.hypervar)

        self.ln_freespec = ceffyl_pta.density[0,:self.Nfreqs]
        
        # saving parameter names
        env_names = [p.name for p in self.hypervar]
        self.param_names = env_names
        
        return
        
    def ln_likelihood(self, x0):
        """
        likelihood function
        """
        # ensure constant values are in the correct place!
        etac = np.zeros(len(self.hyperparams))  # empty array
        etac[self.hypervar_idx] = x0
        etac[self.const_idx] = self.const_values
        
        ## Predict GP
        hc, _, log10h2cf_sigma = gp_utils.hc_from_gp(self.gp_george,
                                                     self.gp, etac)
        log10h2cf_sigma = log10h2cf_sigma[:,1]  # uncertainty on log10h2cf

        ## Convert Zero-Mean to Characteristic Strain Squared
        h2cf = hc**2

        # turn predicted h2cf to psd/T to log10_rho
        psd =  h2cf/(12*np.pi**2 *
                     self.freqs**3 * self.Tspan)
        log10_rho_gp = 0.5*np.log10(psd)[:,None]

        # propogate uncertainty from log10_h2cf to log10_rho
        # propogations cancel to get log10rho_sigma=log10h2cf_sigma/2 !!
        log10rho_sigma = (0.5*log10h2cf_sigma)[:,None]

        # compare GP predicted log10rho to log10rho grid using Gaussian
        ln_gaussian = -0.5 * (((self.rho_grid -
                                log10_rho_gp)/log10rho_sigma)**2 +
                              np.log(2*np.pi*log10rho_sigma**2))
        
        ln_freespec = self.ln_freespec

        # basic integral over rho - need to trapesium rule!
        drho = self.ceffyl_pta.rho_grid[1] - self.ceffyl_pta.rho_grid[0]
        ln_integrand = ln_freespec + ln_gaussian + np.log(drho)
        ln_like = logsumexp(ln_integrand)  # need to vectorise for pulsars

        return ln_like  # return ln gaussian
    
    def ln_likelihood_powerlaw_test(self, x0, sigma=0.01):
        """
        likelihood function - instead of using a GP, a powerlaw is fitted
        with some variance sigma
        """
        # ensure constant values are in the correct place!
        etac = np.zeros(len(self.hyperparams))  # empty array
        etac[self.hypervar_idx] = x0
        etac[self.const_idx] = self.const_values
        
        ## Predict GP
        log10_rho_pl = 0.5*np.log10(gp_priors.powerlaw(self.freqs, *x0,
                                                       components=1))[:,None]
        # uncertainty on log10h2cf
        log10h2cf_sigma = np.repeat(sigma, repeats=self.Nfreqs)

        # propogate uncertainty from log10_h2cf to log10_rho
        # propogations cancel to get log10rho_sigma=log10h2cf_sigma/2 !!
        log10rho_sigma = (0.5*log10h2cf_sigma)[:,None]

        # compare GP predicted log10rho to log10rho grid using Gaussian
        ln_gaussian = -0.5 * (((self.rho_grid -
                                log10_rho_pl)/log10rho_sigma)**2 +
                              np.log(2*np.pi*log10rho_sigma**2))
        
        ln_freespec = self.ln_freespec

        # basic integral over rho - need to trapesium rule!
        drho = self.ceffyl_pta.rho_grid[1] - self.ceffyl_pta.rho_grid[0]
        ln_integrand = ln_freespec + ln_gaussian + np.log(drho)
        ln_like = logsumexp(ln_integrand)  # need to vectorise for pulsars

        return ln_like  # return ln gaussian
    
    def ln_prior(self, x0):
        """
        function to calc total ln_prior
        """
        # separate parameters
        lnprior_eta = np.array([h.get_logpdf(x) for
                                h, x in zip(self.hyperparams, x0)]).sum()
        
        return lnprior_eta #    + lnprior_rho
    
    def initial_samples(self):
        """
        function to get array of initial samples
        """
        #log10_rhos = self.log10_rho.sample()
        etas = [h.sample() for h in self.hypervar]
        
        return np.hstack(etas)


################################################################################
class ceffylGPSampler():
    """
    A class to quickly set-up ceffylGP and sample!
    """
    def __init__(self, trainedGPdir, spectradir, ceffyldir, hyperparams,
                 outdir, emp_dist_dir=None, resume=True, Nfreqs=None,
                 freq_idxs=None, log10_rho_priors=[-10., -5.9], jump=True,
                 test=False, test_sigma=0.01):
        """
        Initialise the class!

        @param trainedGPdir: path to trained GaussProc objects
        @param spectradir: path to original spectra
        @param ceffyldir: path to ceffyl free spectrum representations
        @param hyperparams: list of enterprise.signals.parameter objects
                            NOTE: assumes variables are in order
        @param outdir: directory to save MCMC samples
        @param emp_dist_dir: directory containing list of empirical
                             distributions to aid sampling
        @param resume: flag to resume MCMC sampling
        @param Nfreqs: number of frequencies to fit to
        @param freq_idxs: alternative to Nfreqs - input indexes of frequency
                          bins to fit to
        @param log10_rho_priors: min/max to search over log10_rho -- NEEDS TO
                                 BE AUTOMATED
        @param test: flag to test powerlaw w/ variance, not GPs. Loaded GPs
                     will be ignored\
        @param test_sigma: hard coded std dev to add test powerlaw at each freq
                           if test==True
        """

        # Load trained GPs
        with open(trainedGPdir, "rb") as f:  # load GaussProc objects
            gp_george = pickle.load(f)  # this is not a list of George objects

        spectra = h5py.File(spectradir)  # open spectra file

        # set up list of GP George objects
        gp = gp_utils.set_up_predictions(spectra, gp_george)

        if Nfreqs is None:
            gp_george = list(np.array(gp_george)[freq_idxs])
            gp = list(np.array(gp)[freq_idxs])
        else:
            gp_george = gp_george[:Nfreqs]
            gp = gp[:Nfreqs]

        # set up ceffylGP class
        ceffyl_gp = ceffylGP(ceffyldir, Nfreqs=Nfreqs, hyperparams=hyperparams,
                             gp=gp, gp_george=gp_george, freq_idxs=freq_idxs,
                             log10_rho_priors=log10_rho_priors)
        self.ceffyl_gp = ceffyl_gp

        # parameter groupings for better sampling
        groups = [np.arange(len(ceffyl_gp.param_names))]
        
        for label in ['hard', 'gsmf', 'gpf', 'gmt', 'mmb']:
            if np.any([label in par for par in ceffyl_gp.param_names]):
                groups.append([ceffyl_gp.param_names.index(p) for p in
                               ceffyl_gp.param_names if label in p])

        # set up sampler
        x0 = ceffyl_gp.initial_samples()
        cov = np.identity(len(x0))*0.01

        if test:  # flag to test code against a powerlaw w/ variance
            logl = ceffyl_gp.ln_likelihood_powerlaw_test
        else:
            logl = ceffyl_gp.ln_likelihood
        
        sampler = ptmcmc(len(x0), logl=logl,
                         logp=ceffyl_gp.ln_prior, cov=cov, groups=groups,
                         outDir=outdir, resume=resume)
        
        # load empirical distributions
        if emp_dist_dir is not None:
            with open(emp_dist_dir, 'rb') as f:
                empirical_distr = pickle.load(f)
        else:
            empirical_distr = None

        # add jump proposals for better sampling
        if jump:
            jp = JumpProposal(ceffyl_gp, empirical_distr=empirical_distr)
            sampler.addProposalToCycle(jp.draw_from_prior, 10)
            sampler.addProposalToCycle(jp.draw_from_env_prior, 20)

            if np.any(['hard' in par for par in ceffyl_gp.param_names]):
                sampler.addProposalToCycle(jp.draw_from_hard_prior, 10)

            if np.any(['gsmf' in par for par in ceffyl_gp.param_names]):
                sampler.addProposalToCycle(jp.draw_from_gsmf_prior, 10)

            if np.any(['gpf' in par for par in ceffyl_gp.param_names]):
                sampler.addProposalToCycle(jp.draw_from_gpf_prior, 10)

            if np.any(['gmt' in par for par in ceffyl_gp.param_names]):
                sampler.addProposalToCycle(jp.draw_from_gmt_prior, 10)

            if np.any(['mmb' in par for par in ceffyl_gp.param_names]):
                sampler.addProposalToCycle(jp.draw_from_mmb_prior, 10)

        np.savetxt(outdir+'/pars.txt', ceffyl_gp.param_names, fmt='%s')

        # save sampler
        self.sampler = sampler

        return
    
################################################################################
if __name__ == '__main__':
    """
    Let's test if this code works!
    """
    # paths to data
    trainedGPdir = '../../ng15yr_astro_interp/spec_libraries/circ-01_2023-02-23_01_n1000_s60_r100_f40/trained_gp_circ-01_2023-02-23_01_n1000_s60_r100_f40.pkl'
    spectradir = '../../ng15yr_astro_interp/spec_libraries/circ-01_2023-02-23_01_n1000_s60_r100_f40/sam_lib.hdf5'
    ceffyldir = '../../ng15yr_astro_interp/ceffyl_ng15_multiorf_hdonly'
    emp_dist_dir = '../../ng15yr_astro_interp/ceffyl_ng15_multiorf_hdonly/fs_emp_dist.pkl'

    # path to save MCMC chain
    #outdir = './test/'
    outdir = '../../ng15yr_astro_interp/spec_libraries/circ-01_2023-02-23_01_n1000_s60_r100_f40/test_5f/'

    import sys
    sys.path.append('../../ng15yr_astro_interp/')
    with open(trainedGPdir, 'rb') as f:
        gp = pickle.load(f)  # open directory of trained GPs
    
    # automated parameter set-up
    hp_names = list(gp[0].par_dict.keys())
    hyperparams = []
    for hp in hp_names:
        h = parameter.Uniform(gp[0].par_dict[hp]['min'],
                              gp[0].par_dict[hp]['max'])(hp)
        hyperparams.append(h)

    test_constant = False  # test if parameter.constant function works
    if test_constant:
        hyperparams[-1] = parameter.Constant(1.5)

    Nfreqs = 5  # number of frequencies to fit

    # set up sampler!
    sampler = ceffylGPSampler(trainedGPdir=trainedGPdir, spectradir=spectradir,
                                ceffyldir=ceffyldir, hyperparams=hyperparams,
                                Nfreqs=Nfreqs, outdir=outdir,
                                emp_dist_dir=emp_dist_dir, jump=True)
    
    print('Here are your parameters...\n')
    print(sampler.ceffyl_gp.param_names)
    
    x0 = sampler.ceffyl_gp.initial_samples()
    N = int(1e7)

    #sampler.sampler.sample(x0, N)

    import la_forge.core as co
    import la_forge.diagnostics as dg
    chain = co.Core(outdir)
    #dg.plot_chains(chain, hist=False, save='./trace.png', figsize=(4,10))

    from chainconsumer.chainconsumer import ChainConsumer
    c = ChainConsumer()
    c.add_chain(chain(chain.params[:-4]),
                parameters=chain.params[:-4])
    c.configure(summary=True, smooth=False)
    fig = c.plotter.plot()
    fig.savefig('./testfig.png')
