from ceffyl.ceffyl_gp import ceffylGPSampler
from argparse import ArgumentParser
import pickle
from enterprise.signals import parameter
import numpy as np

# import gp utils
import sys
sys.path.append('/home/lambwg/ng15yr_astro_interp')

# add arguments
parser = ArgumentParser()

parser.add_argument('--Nfreqs', type=int, help='number of frequencies')
parser.add_argument('--gp_pkl', type=str, help='path to trained GP pickles')
parser.add_argument('--var_gp_pkl', type=str,
                    help='path to trained GP pickles trained on variance')
parser.add_argument('--spectra', type=str, default=None,
                    help='path to hdf5 to holodeck spectra')
parser.add_argument('--ceffyldir', type=str, help='path to ceffyl files')
parser.add_argument('--outdir', type=str,
                    help='path to directory to save mcmc')
parser.add_argument('--resume', action='store_true', default=False,
                    help='flag to resume MCMC run')
parser.add_argument('--drop_first', action='store_true', default=False,
                    help='drop the first frequency bin')
parser.add_argument('--Nsamples', type=int, default=int(5e6),
                    help='set number of samples')

args = parser.parse_args()

# open directory of trained GPs
with open(args.gp_pkl, 'rb') as f:
    gp = pickle.load(f)

# automated parameter set-up
hp_names = list(gp[0].par_dict.keys())
hyperparams = []
for hp in hp_names:
    h = parameter.Uniform(gp[0].par_dict[hp]['min'],
                          gp[0].par_dict[hp]['max'])(hp)
    hyperparams.append(h)
    
if args.drop_first:  # freq idx with dropped first freq bin
    freq_idxs = np.arange(1, args.Nfreqs)
    Nfreqs = None
else:
    freq_idxs = None
    Nfreqs = args.Nfreqs

# set up sampler!
sampler = ceffylGPSampler(trainedGP=args.gp_pkl, trained_varGP=args.var_gp_pkl,
                          spectrafile=args.spectra,
                          ceffyldir=args.ceffyldir, hyperparams=hyperparams,
                          Nfreqs=Nfreqs, outdir=args.outdir,
                          resume=args.resume, freq_idxs=freq_idxs)

print('Here are your parameters...\n')
print(sampler.ceffyl_gp.param_names)

x0 = sampler.ceffyl_gp.initial_samples()
N = int(args.Nsamples)

sampler.sampler.sample(x0, N)
