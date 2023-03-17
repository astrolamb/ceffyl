import la_forge.core as co
import la_forge.diagnostics as dg
from emcee.autocorr import integrated_time
import numpy as np
import matplotlib.pyplot as plt

def chain_utils(chaindir=None, corepath=None):
    """
    utility function to plot important info about chains
    """
    # load the chain
    chain = co.Core(chaindir) if chaindir else co.Core(corepath=corepath)

    # list param names
    print(f'These are your {len(chain.params[:-4])} parameters:\n{chain.params[:-4]}\n')

    # plot traceplots
    dg.plot_chains(chain, hist=False)

    # calculate and plot grubin
    Rhat, idx = dg.grubin(chain)
    print(f'Min/max Gelman-Rubin tests: {min(Rhat), max(Rhat)}\n')
    if idx.any():
        print(f'Undersampled parameters: {np.array(chain.params)[idx]}\n')
    dg.plot_grubin(chain)

    # plot histograms of parameters
    dg.plot_chains(chain)

    # calculate and plot ACLs
    acls = np.array([integrated_time(chain(p)) for p in chain.params[:-4]])
    print(f'Min/max Gelman-Rubin tests: {min(acls), max(acls)}\n')
    plt.scatter(np.arange(len(chain.params[:-4])), acls)
    plt.xlabel('Param idx')
    plt.ylabel('ACL')
    plt.title('Autocorrelation length');

    return chain