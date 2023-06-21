###############################################################################
# ceffylGP
# A script to fit trained GPs to PTA free spectrum
# Author: William G. Lamb 2023
###############################################################################

# imports
import numpy as np
import h5py
import pickle
from enterprise.signals import parameter
from enterprise_extensions import sampler
from ceffyl import Ceffyl, models
from PTMCMCSampler.PTMCMCSampler import PTSampler as ptmcmc

try:  # attempt to import holodeck
    from holodeck.gps import gp_utils
except ImportError:
    print('Holodeck is required to run ceffylGP. Please install:\n')
    print('https://github.com/nanograv/holodeck')

from scipy.special import logsumexp
from enterprise.signals import gp_priors
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import Delaunay
###############################################################################


class JumpProposal(object):
    """
    A class to propose jumps for parallel tempered swaps
    Shamelessly copied and modified from
    enterprise_extensions (https://github.com/nanograv/enterprise_extensions/)
    and
    PTMCMCSampler (https://github.com/jellis18/PTMCMCSampler/)
    """
    def __init__(self, ceffylGP):
        """
        Set up some custom jump proposals

        @params ceffylGP - import an initialised ceffylGP class
        """

        # save information as class properties
        self.params = ceffylGP.params  # list of parameter objects
        self.param_names = ceffylGP.param_names  # list of parameter names
        self.hyperparams = ceffylGP.hypervar  # list of gp param names
        self.hypernames = [h.name for h in ceffylGP.hypervar]

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
###############################################################################


class ceffylGP():
    """
    A class to fit GP to PTA free spectrum via ceffyl
    """
    def __init__(self, datadir, hyperparams, gp, gp_george, var_gp,
                 var_gp_george, spectrum=None, Nfreqs=None, freq_idxs=None,
                 log10_rho_priors=[-10, -6]):
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
        self.var_gp = var_gp
        self.var_gp_george = var_gp_george
        
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

        # save rho grid
        # rho_mask = (self.ceffyl_pta.rho_grid > log10_rho_priors[0] and 
        #            self.ceffyl_pta.rho_grid < log10_rho_priors[1])
        rho_grid = self.ceffyl_pta.rho_grid
        
        num_freq = len(freq_idxs) if freq_idxs is not None else Nfreqs
        self.rho_grid = np.repeat([rho_grid], repeats=num_freq,
                                  axis=0)  # save freespec probability grid

        # saving locations of constant hyperparams
        const_idx = np.where(np.array([hasattr(h, 'sample')
                                       for h in hyperparams]) == False)[0]
        const_values = np.array([h.value for h
                                 in np.array(hyperparams)[const_idx]])
        self.const_idx, self.const_values = const_idx, const_values
        
        # locations of variable hyperparams
        self.hypervar_idx = np.where(np.array([hasattr(h, 'sample')
                                               for h
                                               in hyperparams]) == True)[0]
        self.hypervar = np.array(hyperparams)[self.hypervar_idx]

        # saving params that can be sampled (i.e. no constant values)
        self.params = list(self.hypervar)

        if freq_idxs is not None:
            self.ln_freespec = ceffyl_pta.density[0, freq_idxs]
            self.freqs = ceffyl_pta.freqs[freq_idxs]  # save frequencies
        else:
            self.ln_freespec = ceffyl_pta.density[0, :self.Nfreqs]
            self.freqs = ceffyl_pta.freqs[:Nfreqs]  # save frequencies
        
        # saving parameter names
        env_names = [p.name for p in self.hypervar]
        self.param_names = env_names

        # block for spectrum library interpolation
        # FIX ME: apply to specific, chosen freq bins
        # FIX ME: can you optimise interpolation?
        if freq_idxs is not None and spectrum is not None:
            print('cannot use interpolation at the moment with manual freq input')
            
            gwb_spectra = np.array(spectrum['gwb'])[:, :Nfreqs]
            
            # clean the spectra
            nan_ind = np.any(np.isnan(gwb_spectra), axis=(1, 2))
            self.gwb_spectra = gwb_spectra[~nan_ind]

            samples = np.array(spectrum['sample_params'])
            self.samples = samples[~nan_ind]
            
            # create interpolation
            tri = Delaunay(self.samples)
            interpolator = LinearNDInterpolator(tri, self.gwb_spectra)
            self.interpolator = interpolator
        else:
            self.spectrum = None
        
        return
        
    def ln_likelihood(self, x0):
        """
        likelihood function
        """
        # ensure constant values are in the correct place!
        etac = np.zeros(len(self.hyperparams))  # empty array
        etac[self.hypervar_idx] = x0
        etac[self.const_idx] = self.const_values

        # Predict GP
        hc, _, log10h2cf_sigma = gp_utils.hc_from_gp(self.gp_george,
                                                     self.gp,
                                                     self.var_gp_george,
                                                     self.var_gp, etac)
        log10h2cf_sigma = log10h2cf_sigma[:, 1]  # uncertainty on log10h2cf

        # Convert Zero-Mean to Characteristic Strain Squared
        h2cf = hc**2

        # turn predicted h2cf to psd/T to log10_rho
        psd = h2cf/(12*np.pi**2 *
                     self.freqs**3 * self.Tspan)
        log10_rho_gp = 0.5*np.log10(psd)[:, None]

        # propogate uncertainty from log10_h2cf to log10_rho
        # propogations cancel to get log10rho_sigma=log10h2cf_sigma/2 !!
        log10rho_sigma = (0.5*log10h2cf_sigma)[:, None]

        # compare GP predicted log10rho to log10rho grid using Gaussian
        ln_gaussian = -0.5 * (((self.rho_grid -
                                log10_rho_gp)/log10rho_sigma)**2 +
                              np.log(2*np.pi*log10rho_sigma**2))
        
        ln_freespec = self.ln_freespec

        # numerical integration using the trapezium rule
        dlog10rho = self.ceffyl_pta.rho_grid[1] - self.ceffyl_pta.rho_grid[0]
        ln_integrand = ln_freespec + ln_gaussian
        ln_integrand[1:-1] += np.log(2)
        ln_like = logsumexp(ln_integrand, axis=1, b=0.5*dlog10rho)

        return np.sum(ln_like)  # return ln gaussian
    
    def ln_likelihood_powerlaw_test(self, x0, sigma=0.01):
        """
        likelihood function - instead of using a GP, a powerlaw is fitted
        with some variance sigma
        """
        # ensure constant values are in the correct place!
        etac = np.zeros(len(self.hyperparams))  # empty array
        etac[self.hypervar_idx] = x0
        etac[self.const_idx] = self.const_values
        
        # Predict GP
        log10_rho_pl = 0.5*np.log10(gp_priors.powerlaw(self.freqs, *x0,
                                                       components=1))[:, None]
        # uncertainty on log10h2cf
        log10h2cf_sigma = np.repeat(sigma, repeats=self.Nfreqs)

        # propogate uncertainty from log10_h2cf to log10_rho
        # propogations cancel to get log10rho_sigma=log10h2cf_sigma/2 !!
        log10rho_sigma = (0.5*log10h2cf_sigma)[:, None]

        # compare GP predicted log10rho to log10rho grid using Gaussian
        ln_gaussian = -0.5 * (((self.rho_grid -
                                log10_rho_pl)/log10rho_sigma)**2 +
                              np.log(2*np.pi*log10rho_sigma**2))
        
        ln_freespec = self.ln_freespec

        # basic integral over rho - need to trapesium rule!
        drho = self.ceffyl_pta.rho_grid[1] - self.ceffyl_pta.rho_grid[0]
        ln_integrand = ln_freespec + ln_gaussian + np.log(drho)
        ln_like = logsumexp(ln_integrand, axis=1)

        return np.sum(ln_like)  # return ln gaussian

    def holospectrum_lnlikelihood(self, x0):
        """
        function to fit holodeck simulations quickly
        w/o GPs. Not as accurate, but rather fast!
        """
        # ensure constant values are in the correct place!
        xs = np.zeros(len(self.hyperparams))  # empty array
        xs[self.hypervar_idx] = x0
        xs[self.const_idx] = self.const_values
        
        interp = self.interpolator(x0)
        
        # find mean, sigma of spectra at these points
        #hc = np.median(self.gwb_spectra[idx], axis=1)
        #sigma_hc = np.std(self.gwb_spectra[idx], axis=1)

        hc = np.median(interp, axis=2)
        sigma_hc = np.std(interp, axis=2)
        
        # turn predicted h2cf to psd/T to log10_rho
        psd =  hc**2/(12*np.pi**2 * self.freqs**3 * self.Tspan)
        log10_rho_gp = 0.5*np.log10(psd).T
        
        # propogate uncertainty from hc to log10_rho
        log10rho_sigma = (sigma_hc/(hc*np.log(10))).T

        # compare GP predicted log10rho to log10rho grid using Gaussian
        ln_gaussian = -0.5 * (((self.rho_grid -
                                log10_rho_gp)/log10rho_sigma)**2 +
                              np.log(2*np.pi*log10rho_sigma**2))

        ln_freespec = self.ln_freespec

        # basic integral over rho - need to trapesium rule!
        drho = self.ceffyl_pta.rho_grid[1] - self.ceffyl_pta.rho_grid[0]
        ln_integrand = ln_freespec + ln_gaussian + np.log(drho)
        ln_like = logsumexp(ln_integrand, axis=1)  # need to vectorise for pulsars
        
        return np.sum(ln_like)  # return ln gaussian
    
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
    def __init__(self, trainedGP, trained_varGP, ceffyldir,
                 hyperparams, outdir, spectrafile=None,
                 resume=True, Nfreqs=None, freq_idxs=None,
                 log10_rho_priors=[-10., -5.9], jump=True,
                 analysis_type='gp', test_sigma=0.01):
        """
        Initialise the class!

        @param trainedGPdir: path to trained GaussProc objects
        @param spectradir: path to original spectra
        @param ceffyldir: path to ceffyl free spectrum representations
        @param hyperparams: list of enterprise.signals.parameter objects
                            NOTE: assumes variables are in order
        @param outdir: directory to save MCMC samples
        @param resume: flag to resume MCMC sampling
        @param Nfreqs: number of frequencies to fit to
        @param freq_idxs: alternative to Nfreqs - input indexes of frequency
                          bins to fit to
        @param log10_rho_priors: min/max to search over log10_rho -- NEEDS TO
                                 BE AUTOMATED
        @param analysis_type: kwarg to change logL
                              - 'gp' will use trained GPs
                              - 'holo_spectra' will approximate using spectra
                                straight from holodeck
                              - 'test' will use powerlaw w/ variance
        @param test_sigma: hard coded std dev to add test powerlaw at each freq
                           if test==True
        """

        # load spectra is file provided
        if spectrafile is not None:
            spectra = h5py.File(spectrafile)  # open spectra file
        else:
            spectra = None  # not needed for most runs

        # Load trained GPs
        if trainedGP is not None:
            with open(trainedGP, "rb") as f:  # load GaussProc objects
                gp_george = pickle.load(f)  # this is not a list of George objects

            # set up list of GaussProc objects
            gp = gp_utils.set_up_predictions(spectra, gp_george)
        else:
            gp, gp_george = None, None

        if gp is not None and gp_george is not None:
            if Nfreqs is None:
                gp_george = np.array(gp_george)[freq_idxs].tolist()
                
                gp_list = []  # GPGeorge behaves weirdly with np arrays
                for idx in freq_idxs:
                    gp_list.append(gp[idx])
                gp = gp_list
            else:
                gp_george = gp_george[:Nfreqs]
                gp = gp[:Nfreqs]

        if trained_varGP is not None:
            with open(trained_varGP, "rb") as f:  # load GaussProc objects
                var_gp_george = pickle.load(f)  # this is not a list of George objects

            # set up list of GP George objects
            var_gp = gp_utils.set_up_predictions(spectra, var_gp_george)
        else:
            var_gp, var_gp_george = None, None

        if var_gp is not None and var_gp_george is not None:
            if Nfreqs is None:
                var_gp_george = list(np.array(var_gp_george)[freq_idxs])
                
                var_gp_list = []  # GPGeorge behave weirdly with np arrays
                for idx in freq_idxs:
                    var_gp_list.append(var_gp[idx])
                var_gp = var_gp_list
            else:
                var_gp_george = var_gp_george[:Nfreqs]
                var_gp = var_gp[:Nfreqs]

        # set up ceffylGP class
        ceffyl_gp = ceffylGP(ceffyldir, Nfreqs=Nfreqs, hyperparams=hyperparams,
                             gp=gp, gp_george=gp_george, var_gp=var_gp,
                             var_gp_george=var_gp_george, freq_idxs=freq_idxs,
                             log10_rho_priors=log10_rho_priors,
                             spectrum=spectra)
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

        if analysis_type == 'test':  # flag to test code against a powerlaw w/ variance
            logl = ceffyl_gp.ln_likelihood_powerlaw_test
        elif analysis_type == 'holo_spectra':
            logl = ceffyl_gp.holospectrum_lnlikelihood
        elif analysis_type == 'gp':
            logl = ceffyl_gp.ln_likelihood
        else:
            print("Please choose between 'test', 'holo_spectra', and 'gp'\n")
            return
        
        sampler = ptmcmc(len(x0), logl=logl,
                         logp=ceffyl_gp.ln_prior, cov=cov, groups=groups,
                         outDir=outdir, resume=resume)

        # add jump proposals for better sampling
        if jump:
            jp = JumpProposal(ceffyl_gp)
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

    # path to save MCMC chain
    #outdir = './test/'
    outdir = '../../ng15yr_astro_interp/spec_libraries/circ-01_2023-02-23_01_n1000_s60_r100_f40/test_5f_bin/'

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
                              analysis_type='holo_spectra', jump=True)
    
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
    fig.savefig(outdir + '/testfig.png')
