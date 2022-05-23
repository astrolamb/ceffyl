import numpy as np


"""
Some code to calculate the hellinger distance of a 2D distributions, with
more options to come soon!
"""


def hellinger(gamma_chain1, log10A_chain1, gamma_chain2, log10A_chain2, Nbins):
    """
    A function to calculate hellinger distance for two analyses with log10_A
    and gamma

    @param gamma_chain1: chain of gamma values for first analysis
    @param log10A_chain1: chain of log10_A values for first analysis
    @param gamma_chain2: chain of gamma values for first analysis
    @param log10A_chain2: chain of log10_A values for second analysis
    @param Nbins: number of bins to histogram

    @return hellinger: the Hellinger metric between the analyses
    """
    # find min of gamma, log10_A chains
    min_gamma = min(gamma_chain1.min(), gamma_chain2.min())
    min_log10A = min(log10A_chain1.min(), log10A_chain2.min())

    # find max of gamma, log10_A chains
    max_gamma = max(gamma_chain1.max(), gamma_chain2.max())
    max_log10A = max(log10A_chain1.max(), log10A_chain2.max())

    # bins by limits
    gamma_bins = np.linspace(min_gamma, max_gamma, Nbins)
    log10A_bins = np.linspace(min_log10A, max_log10A, Nbins)

    # infinitessimals
    dg = (max_gamma-min_gamma)/Nbins
    dA = (max_log10A-min_log10A)/Nbins

    # normalised probability distributions
    P = np.histogram2d(x=gamma_chain1, y=log10A_chain1,
                       bins=(gamma_bins, log10A_bins), density=True)[0]*dg*dA
    Q = np.histogram2d(x=gamma_chain2, y=log10A_chain2,
                       bins=(gamma_bins, log10A_bins), density=True)[0]*dg*dA

    # hellinger
    hellinger = np.sqrt(np.sum((np.sqrt(P)-np.sqrt(Q))**2))/np.sqrt(2)

    return hellinger
