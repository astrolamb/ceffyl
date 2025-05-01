import la_forge.core as co
import la_forge.diagnostics as dg
from emcee.autocorr import integrated_time
import numpy as np
import matplotlib.pyplot as plt

def read_data(chaindir=None, corepath=None, core=None, pars=None,
              burn=0.25):
    """
    function to read_data and return a np array of chosen pars
    """
    # load the chain
    if chaindir:
        core = co.Core(chaindir, burn=burn)
    elif corepath:
        core = co.Core(corepath=corepath, burn=burn)
    elif core:
        core = core
    else:
        print('Gimme some data to chew on')
        return
    
    if pars is None:  # set params is none
        pars = core.params[:-4]
    
    return core, pars

def print_info(core):
    """
    print chain diagnostics from cores
    """
    # calculate and plot ACLs
    acls = np.array([integrated_time(core(p), quiet=True)
                     for p in core.params[:-4]])
    
    # calculate grubin
    Rhat, idx = dg.grubin(core)
    print(f'Min/max Gelman-Rubin tests: {np.min(Rhat), np.max(Rhat)}\n')
    
    dg.plot_grubin(core)
    
    if idx.any():
        print(f'Undersampled parameters: {np.array(core.params)[idx]}\n')

    print(f'Min/max autocorrelation lengths: {np.min(acls), np.max(acls)}\n')
    plt.scatter(np.arange(len(core.params[:-4])), acls)

    plt.xlabel('Param idx')
    plt.ylabel('ACL')
    plt.title('Autocorrelation length');
    
    return


def chain_utils(chaindir=None, corepath=None,
                core=None, pars=None, burn=0.25):
    """
    utility function to plot important info about chains
    """
    
    # read data
    core, pars = read_data(chaindir=chaindir, corepath=corepath,
                            core=core, pars=pars, burn=burn)

    # list param names
    print(f'These are your {len(pars)} parameters:\n{pars}\n')

    dg.plot_chains(core, hist=False, pars=pars)  # plot traceplots

    #dg.plot_chains(core, pars=pars)  # plot histograms of parameters
    
    print_info(core)  # print chain diagnostics

    return core
