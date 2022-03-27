# imports
import numpy as np
import pandas as pd
import os
from os.path import exists

"""
This file contains classes to compress chain txt files from PTMCMCSampler
into .npy/.npz files for use in ceffyl
"""

metadata = ['logL', 'unweighted_logPosterior',
            'MCMC_acceptance_rate',
            'interchain_transitions_acceptance_rate']


class compress_PTA_chain():
    """
    Class to compress a single MCMC chain as a npy file

    Inputs:
    :str: datadir      = location of directory containing chain and parameter
                         file
    :str: chainfile    = name of chain file. Default: chain_1.0.txt. Note,
                         chain often called chain_1.txt too
    :str or list: pars = either a list of paremeters for the chain, or a string
                         corresponding to the name of a file containing par
                         names
    """

    def __init__(self, datadir, chainfile='chain_1.0.txt', pars='pars.txt'):
        """
        A function to load chains and file them as a dictionary of dataframes
        """
        # need - which region of chain to take e.g. lop off the end

        # read chains
        print('Loading chain...')

        if isinstance(pars, list):
            parlist = pars

        elif exists(datadir+pars):
            parlist = np.append(np.loadtxt(datadir+pars,
                                           dtype=np.unicode_), metadata)

        else:
            print('Parameter file not found!')
            return

        if exists(datadir+chainfile):
            chainpath = (datadir + chainfile)
            self.chainpath = chainpath  # ready to delete...
            chain = pd.read_csv(chainpath, names=parlist,
                                usecols=parlist, delim_whitespace=True)

            self.chain = chain

            return

        else:
            print('Chain file not found!')
            return

    def burn(self, burnin=0.25, burnout=None, thin=None):
        """
        function to let it burn
        """

        # size of chain
        size = self.chain.shape[0]

        # burn in
        if (0 < burnin and burnin < 1):  # if burnin is a fraction
            burn = int(burnin*size)

        elif isinstance(burnin, int):  # if burnin is an int
            if burnin > size:
                print("Burn-in is bigger than size of chain! No burning")

            else:
                burn = burnin

        else:
            burn = 0

        self.chain = self.chain[burn:]
        newsize = self.chain.shape[0]  # new size of chain

        # burnout
        if burnout is not None:
            if (0 <= burnout and burnout < 1):  # if burnout is a fraction
                burn = int(burnout*size)

            elif isinstance(burnout, int):  # if burnout is an int
                if burnout > newsize:
                    print("Burn-out is bigger than size of chain! No burning")

                else:
                    burn = burnout

            else:
                burn = newsize

            self.chain = self.chain[:-burn]

        # code to thin chain with common thinning value
        if thin is not None:
            if not isinstance(thin, int):
                print('Please input an int!')
                return

            self.chain = self.chain[::thin]

        return

    def compress_chain(self, outfile='compressed_psrs.npz',
                       delete_OG_chain=False, name='freespec'):
        """
        function to save pulsar chain record array as binary npy file
        """

        np.savez_compressed(outfile, **dict(name=self.chain.to_records()))

        if delete_OG_chain:
            self.delete_chainfile()

        return

    def get_chain(self):
        """
        return the chain
        """
        return self.chain

    def delete_chainfile(self):
        """
        delete original chain
        """
        print(f'Deleting {self.chainpath}')
        os.system(f'rm -f {self.chainpath}')


class compress_spsrs_chains():
    """
    class to read data and compress collection of spsrs chains
    """

    def __init__(self, dirlist, pulsarlist,
                 chainfile='chain_1.0.txt', pars='pars.txt'):
        """
        A function to load chains and file them as a dictionary of dataframes
        """
        # need - which region of chain to take e.g. lop off the end

        self.chainpaths = []
        self.psrchains = dict()
        for psr, psrdir in zip(pulsarlist, dirlist):
            # read chains
            print(f'Loading psr {psr}')

            if isinstance(pars, list):
                parlist = pars

            elif exists(psrdir+'/'+pars):
                parlist = np.append(np.loadtxt(psrdir+'/'+pars,
                                               dtype=np.unicode_), metadata)

            else:
                print('Parameter file not found!')
                return

            chainpath = psrdir+'/'+chainfile
            if exists(chainpath):
                self.chainpaths.append(chainpath)  # ready to delete...
                chain = pd.read_csv(chainpath, names=parlist,
                                    usecols=parlist, delim_whitespace=True)

                self.psrchains[psr] = chain.to_records()

            else:
                print(f'{chainfile} not found!')
                return

    def burn(self, burnin=0.25, burnout=None, thin=None):
        """
        function to let it burn
        """

        for ii, psr in enumerate(self.psrchains):

            chain = self.psrchains[psr]

            # size of chain
            size = chain.shape[0]

            # burn in
            if (0 < burnin and burnin < 1):  # if burnin is a fraction
                burn = int(burnin*size)

            elif isinstance(burnin, int):  # if burnout is an int
                if burnin > size:
                    print("Burn-in is bigger than size of chain! No burning")

                else:
                    burn = burnin

            # if you have a list of different burns for different psrs
            elif isinstance(burnin, list):
                burn = burnin[ii]

            else:
                burn = 0

            chain = chain[burn:]
            newsize = self.psrchains[psr].shape[0]  # new size of chain

            # burnout
            if burnout is not None:
                if (0 < burnout and burnout < 1):  # if burnout is a fraction
                    burn = int(burnout*size)

                elif isinstance(burnout, int):  # if burnout is an int
                    if burnout > newsize:
                        print("Burn-out is bigger than size of chain! " +
                              "No burning")

                    else:
                        burn = burnout

                # if you have a list of different burns for different psrs
                elif isinstance(burnout, list):
                    burn = burnout[ii]

                else:
                    burn = newsize

                chain = chain[:-burn]

            # code to thin chain with common thinning value
            if thin:

                if isinstance(thin, list):
                    thin = thin[ii]

                elif not isinstance(thin, int):
                    print('Please input an int!')
                    return

                chain = chain[::thin]

            self.psrchains[psr] = chain

        return

    def compress_chains(self, outfile='compressed_psrs.npz',
                        delete_OG_chains=False):
        """
        function to save pulsar chains as compressed npz files

        arrays saved as series of Nparam arrays for each psr
        """

        np.savez_compressed(outfile, **self.psrchains)

        if delete_OG_chains:
            self.delete_chainfiles()

        return

    def get_chains(self):
        """
        return the chain
        """
        return self.psrchains

    def delete_chainfiles(self):
        """
        delete original chain
        """

        for psrchain in self.chainpaths:
            print(f'Deleting {psrchain}')
            os.system(f'rm -f {psrchain}')

        return
