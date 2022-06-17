"""
Some code to calculate the hellinger distance of a 2D distributions, with
more options to come soon!
"""
import numpy as np


def hellinger(chain1, chain2, Nbins=100):
    """
    A function to calculate hellinger distance for two analyses with log10_A
    and gamma

    @param chain1: n-dim chain (array-like)
    @param chain2: n-dim chain (array-like)

    Note: expect chain1.shape == chain2.shape and chain
          to be 2D. E.g. if comparing two 1D arrays,
          change them into a (N, 1) array using
          numpy broadcasting

    @return hellinger: the Hellinger metric between the analyses
    """
    if chain1.ndim == 1 or chain2.ndim == 1:
        raise IndexError('Ensure chain is two-dimensional')

    col = chain1.shape[1]
    if chain2.shape[1] != col:
        raise ValueError('Only chains with the same ' +
                         'number of columns can be used')

    # find min, max of both chains
    minbins = [min(chain1[:, ii].min(), chain2[:, ii].min())
               for ii in range(col)]
    maxbins = [max(chain1[:, ii].max(), chain2[:, ii].max())
               for ii in range(col)]

    # bins by limits
    bins = [np.linspace(minbin, maxbin, Nbins)
            for minbin, maxbin in zip(minbins, maxbins)]
    dx = np.prod([b[1]-b[0] for b in bins])  # infinitessimals

    # normalised probability distributions
    P = np.histogramdd(chain1, bins=bins, density=True)[0]*dx
    Q = np.histogramdd(chain2, bins=bins, density=True)[0]*dx

    # hellinger
    hellinger = np.sqrt(np.sum((np.sqrt(P)-np.sqrt(Q))**2))/np.sqrt(2)

    return hellinger
