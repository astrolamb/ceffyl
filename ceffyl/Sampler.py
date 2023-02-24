import numpy as np
import pickle
import os
from PTMCMCSampler.PTMCMCSampler import PTSampler as ptmcmc
from enterprise_extensions.empirical_distr import (EmpiricalDistribution1D,
                                                   EmpiricalDistribution1DKDE,
                                                   EmpiricalDistribution2D,
                                                   EmpiricalDistribution2DKDE)


class JumpProposal(object):
    """
    A class to propose jumps for parallel tempered swaps

    Shamelessly copied and modified from

    enterprise_extensions (https://github.com/nanograv/enterprise_extensions/)
    and
    PTMCMCSampler (https://github.com/jellis18/PTMCMCSampler/)
    """
    def __init__(self, signals, empirical_distr=None, save_ext_dists=False,
                 outdir='chains'):
        """
        Set up some custom jump proposals

        @params signals: list of signals for GFL
        """

        # save information as class properties
        self.params = []  # list of parameter objects
        self.param_names = []  # list of parameter names
        self.red_names = []  # list of irn parameter names
        self.gw_names = []  # list of common process parameter names
        self.empirical_distr = empirical_distr  # emp dists

        # loop through signals and save info
        for s in signals:
            self.params.extend(s.params)
            self.param_names.extend(s.param_names)

            if s.CP:  # if signal is a CP, save names of params in this signal
                self.gw_names.extend(s.param_names)
            else:  # else, save different list for comparison purposes
                self.red_names.extend(s.param_names)

        # parameter indices map
        self.pimap = {}
        for ct, p in enumerate(self.param_names):
            self.pimap[p] = ct

        if self.empirical_distr is not None:
            # only save the empirical distributions for
            # parameters that are in the model
            mask = []
            for idx, d in enumerate(self.empirical_distr):
                if d.ndim == 1:
                    if d.param_name in self.param_names:
                        mask.append(idx)
                else:
                    if (d.param_names[0] in self.param_names and
                            d.param_names[1] in self.param_names):
                        mask.append(idx)

            if len(mask) >= 1:
                self.empirical_distr = [self.empirical_distr[m] for m in mask]
                # extend empirical_distr here:
                print('Extending empirical distributions to priors...\n')
                self.empirical_distr = self.extend_emp_dists(
                                          self.empirical_distr,
                                          npoints=100_000,
                                          save_ext_dists=save_ext_dists,
                                          outdir=outdir)
            else:
                self.empirical_distr = None

    def extend_emp_dists(self, emp_dists, npoints=100_000,
                         save_ext_dists=False, outdir='chains'):
        """
        Code to include empirical distributions for faster convergence of
        red noise parameters
        """
        new_emp_dists = []
        modified = False  # check if anything was changed

        for emp_dist in emp_dists:
            if (isinstance(emp_dist, EmpiricalDistribution2D) or
                    isinstance(emp_dist, EmpiricalDistribution2DKDE)):

                # check if we need to extend the distribution
                prior_ok = True
                for ii, (param, nbins) in enumerate(zip(emp_dist.param_names,
                                                        emp_dist._Nbins)):

                    # skip if one of the parameters isn't in our PTA object
                    if param not in self.param_names:
                        continue

                    # check 2 conditions on both params to make sure
                    # that they cover their priors
                    # skip if emp dist already covers the prior
                    param_idx = self.param_names.index(param)
                    prior_min = self.params[param_idx].prior._defaults['pmin']
                    prior_max = self.params[param_idx].prior._defaults['pmax']

                    # no need to extend if hist edges are already prior min/max
                    if isinstance(emp_dist, EmpiricalDistribution2D):
                        if not(emp_dist._edges[ii][0] == prior_min and
                               emp_dist._edges[ii][-1] == prior_max):

                            prior_ok = False
                            continue

                    elif isinstance(emp_dist, EmpiricalDistribution2DKDE):
                        if not(emp_dist.minvals[ii] == prior_min and
                               emp_dist.maxvals[ii] == prior_max):

                            prior_ok = False
                            continue

                if prior_ok:
                    new_emp_dists.append(emp_dist)
                    continue

                modified = True
                samples = np.zeros((npoints, emp_dist.draw().shape[0]))
                for ii in range(npoints):  # generate samples from old emp dist
                    samples[ii] = emp_dist.draw()

                new_bins, minvals, maxvals, idxs_to_remove = [], [], [], []

                for ii, (param, nbins) in enumerate(zip(emp_dist.param_names,
                                                        emp_dist._Nbins)):
                    param_idx = self.param_names.index(param)
                    prior_min = self.params[param_idx].prior._defaults['pmin']
                    prior_max = self.params[param_idx].prior._defaults['pmax']

                    # drop samples that are outside the prior range
                    # (in case prior is smaller than samples)
                    if isinstance(emp_dist, EmpiricalDistribution2D):
                        samples[(samples[:, ii] < prior_min) |
                                (samples[:, ii] > prior_max), ii] = -np.inf

                    elif isinstance(emp_dist, EmpiricalDistribution2DKDE):
                        idxs_to_remove.extend(np.arange(npoints)
                                              [(samples[:, ii] < prior_min) |
                                               (samples[:, ii] > prior_max)])
                        minvals.append(prior_min)
                        maxvals.append(prior_max)

                    # new distribution with more bins this time to extend it
                    # all the way out in same style as above.
                    new_bins.append(np.linspace(prior_min, prior_max,
                                                nbins + 40))

                samples = np.delete(samples, idxs_to_remove, axis=0)
                if isinstance(emp_dist, EmpiricalDistribution2D):
                    new_emp = EmpiricalDistribution2D(emp_dist.param_names,
                                                      samples.T, new_bins)

                elif isinstance(emp_dist, EmpiricalDistribution2DKDE):
                    # new distribution with more bins this time to extend it
                    # all the way out in same style as above.
                    bandwidth = emp_dist.bandwidth
                    new_emp = EmpiricalDistribution2DKDE(emp_dist.param_names,
                                                         samples.T,
                                                         minvals=minvals,
                                                         maxvals=maxvals,
                                                         nbins=nbins+40,
                                                         bandwidth=bandwidth)
                new_emp_dists.append(new_emp)

            elif (isinstance(emp_dist, EmpiricalDistribution1D) or
                  isinstance(emp_dist, EmpiricalDistribution1DKDE)):

                if emp_dist.param_name not in self.param_names:
                    continue

                param_idx = self.param_names.index(emp_dist.param_name)
                prior_min = self.params[param_idx].prior._defaults['pmin']
                prior_max = self.params[param_idx].prior._defaults['pmax']

                # check 2 conditions on param to make sure that it covers the
                # prior skip if emp dist already covers the prior
                if isinstance(emp_dist, EmpiricalDistribution1D):
                    if (emp_dist._edges[0] == prior_min and
                            emp_dist._edges[-1] == prior_max):
                        new_emp_dists.append(emp_dist)
                        continue

                elif isinstance(emp_dist, EmpiricalDistribution1DKDE):
                    if (emp_dist.minval == prior_min and
                            emp_dist.maxval == prior_max):
                        new_emp_dists.append(emp_dist)
                        continue

                modified = True
                samples = np.zeros((npoints, 1))
                for ii in range(npoints):  # generate samples from old emp dist
                    samples[ii] = emp_dist.draw()
                new_bins = []
                idxs_to_remove = []

                # drop samples that are outside the prior range
                # (in case prior is smaller than samples)
                if isinstance(emp_dist, EmpiricalDistribution1D):
                    samples[(samples < prior_min) |
                            (samples > prior_max)] = -np.inf

                elif isinstance(emp_dist, EmpiricalDistribution1DKDE):
                    idxs_to_remove.extend(np.arange(npoints)
                                          [(samples.squeeze() < prior_min) |
                                           (samples.squeeze() > prior_max)])

                samples = np.delete(samples, idxs_to_remove, axis=0)
                new_bins = np.linspace(prior_min, prior_max,
                                       emp_dist._Nbins + 40)
                if isinstance(emp_dist, EmpiricalDistribution1D):
                    new_emp = EmpiricalDistribution1D(emp_dist.param_name,
                                                      samples, new_bins)
                elif isinstance(emp_dist, EmpiricalDistribution1DKDE):
                    bandwidth = emp_dist.bandwidth
                    new_emp = EmpiricalDistribution1DKDE(emp_dist.param_name,
                                                         samples,
                                                         minval=prior_min,
                                                         maxval=prior_max,
                                                         bandwidth=bandwidth)
                new_emp_dists.append(new_emp)

            else:
                print('Unable to extend class of unknown type to the edges ' +
                      'of the priors.')
                new_emp_dists.append(emp_dist)
                continue

            # if user wants to save them, and they have been modified...
            if save_ext_dists and modified:
                pickle.dump(new_emp_dists, outdir + 'new_emp_dists.pkl')

        return new_emp_dists

    def draw_from_prior(self, x, iter, beta):
        """
        Draw values from prior for jump

        @param x: array of proposed parameter values
        @param iter: iteration of sampler
        @param beta: inverse temperature of chain

        @return: q: New position in parameter space
        @return: lqxy: log forward-backward jump probability
        """

        q = x.copy()
        lqxy = 0

        p = np.random.choice(self.params)
        pidx = self.params.index(p)

        # sample this parameter
        rand = p.sample()

        # change just one param
        if type(rand) is np.ndarray:
            subparams = [pn for pn in self.param_names if p.name in pn]
            psubp = np.random.choice(subparams)
            psubidx = subparams.index(psubp)
            pidx = self.param_names.index(psubp)
            rand = rand[psubidx]

        # sample this parameter
        q[pidx] = rand

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

    def draw_from_gwb_priors(self, x, iter, beta):
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
        #p_name = np.random.choice(self.gw_names)
        #pidx = self.param_names.index(p_name)
        #p = self.params[pidx]

        p = np.random.choice(self.params)
        pidx = self.params.index(p)

        # sample this parameter
        rand = p.sample()

        # change just one param
        if type(rand) is np.ndarray:
            subparams = [pn for pn in self.param_names if p.name in pn
                         and 'gw' in pn]
            psubp = np.random.choice(subparams)
            psubidx = subparams.index(psubp)
            pidx = self.param_names.index(psubp)
            rand = rand[psubidx]

        # sample this parameter
        q[pidx] = rand

        # forward-backward jump probability
        lqxy = p.get_logpdf(x[pidx] - q[pidx])

        return q, float(lqxy)

    def draw_from_empirical_distr(self, x, iter, beta):
        """
        Prior draw from empirical distributions

        @param x: array of proposed parameter values
        @param iter: iteration of sampler
        @param beta: inverse temperature of chain

        @return: q: New position in parameter space
        @return: lqxy: log forward-backward jump probability
        """
        q = x.copy()
        lqxy = 0

        if self.empirical_distr is not None:

            # randomly choose one of the empirical distributions
            distr_idx = np.random.randint(0, len(self.empirical_distr))

            if self.empirical_distr[distr_idx].ndim == 1:

                idx = self.param_names.index(
                        self.empirical_distr[distr_idx].param_name)
                q[idx] = self.empirical_distr[distr_idx].draw()

                lqxy = (self.empirical_distr[distr_idx].logprob(x[idx]) -
                        self.empirical_distr[distr_idx].logprob(q[idx]))

                dist = self.empirical_distr[distr_idx]
                # if we fall outside the emp distr support
                # pull from prior instead
                if x[idx] < dist._edges[0] or x[idx] > dist._edges[-1]:
                    q, lqxy = self.draw_from_prior(x, iter, beta)

            else:
                dist = self.empirical_distr[distr_idx]
                oldsample = [x[self.param_names.index(p)]
                             for p in dist.param_names]
                newsample = dist.draw()

                lqxy = (dist.logprob(oldsample) - dist.logprob(newsample))

                for p, n in zip(dist.param_names, newsample):
                    q[self.param_names.index(p)] = n

                # if we fall outside the emp distr support
                # pull from prior instead
                for ii in range(len(oldsample)):
                    if (oldsample[ii] < dist._edges[ii][0] or
                            oldsample[ii] > dist._edges[ii][-1]):
                        q, lqxy = self.draw_from_prior(x, iter, beta)

        return q, float(lqxy)


def setup_sampler(ceffyl, outdir, logL, logp, resume=True, jump=True,
                  groups=None, loglkwargs={}, logpkwargs={}, ptmcmc_kwargs={},
                  empirical_distr=None,  save_ext_dists=False):
    """
    Method to setup sampler

    // Inputs //
    @params ceffyl: ceffyl PTA object
    @params outdir: Path to directory to save MCMC chain
    @params logL: Log likelihood function for the MCMC
    @params logp: Log prior function for the MCMC. This should be a prior
                    transform function if nested=True
    @params resume: Flag to toggle option to resume MCMC run from a
                    previous run
    @params jump: Flag to use jump proposals in parallel tempering
    @params groups: indices for which to perform adaptive jumps
    @param loglkwargs: additional kwargs for log likelihood
    @param logpkwargs: additional kwargs for log prior
    @param ptmcmc_kwargs: additional kwargs for PTMCMCSampler
    @param empirical_distr: add empirical distributions to jump proposals
    @param save_ext_dists: flag to save empirical distributions

    @return sampler: initialised PTMCMC sampler
    """

    # initial jump covariance matrix
    if os.path.exists(outdir+'/cov.npy'):
        cov = np.load(outdir+'/cov.npy')
    else:
        cov = np.diag(np.ones(ceffyl.ndim) * 0.1**2)

    # group params for PT swaps
    if groups is None:
        groups = [list(np.arange(0, ceffyl.ndim))]

        # make a group for each signal, with all non-global parameters
        for s in ceffyl.signals:
            groups.extend(s.pmap)

            if s.CP:  # visit GW signals x5 more often
                [groups.extend(s.pmap) for ii in range(5)]
            else:  # group irn by psr - assumes irn added first
                groups.extend(s.psrparam_idxs)

    # sampler
    sampler = ptmcmc(ceffyl.ndim, logL, logp, cov,
                     outDir=outdir, resume=resume,
                     loglkwargs=loglkwargs, logpkwargs=logpkwargs,
                     groups=groups, **ptmcmc_kwargs)

    # save parameter names
    np.savetxt(outdir+'/pars.txt', ceffyl.param_names, fmt='%s')

    # PT swap jump proposals
    if jump:
        jp = JumpProposal(ceffyl.signals, empirical_distr=empirical_distr,
                          save_ext_dists=save_ext_dists, outdir=outdir)
        sampler.jp = jp

        # always add draw from prior
        sampler.addProposalToCycle(jp.draw_from_prior, 5)

        # flags to automatically add prior draws given certain signals
        red_noise, gw_signal = False, False
        for s in ceffyl.signals:
            if s.CP:
                gw_signal = True
            else:
                red_noise = True

        # Red noise prior draw
        if red_noise:
            print('Adding red noise prior draws...\n')
            sampler.addProposalToCycle(jp.draw_from_red_prior, 50)

        # GWB uniform distribution draw
        if gw_signal:
            print('Adding GWB uniform distribution draws...\n')
            sampler.addProposalToCycle(jp.draw_from_gwb_priors, 10)

        # try adding empirical proposals
        if empirical_distr is not None:
            print('Attempting to add empirical proposals...\n')
            sampler.addProposalToCycle(jp.draw_from_empirical_distr, 10)

    return sampler
