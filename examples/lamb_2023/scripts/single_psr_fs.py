# Script to run free spectrum on individual pulsars
# William G. Lamb 2022

import glob
from enterprise.pulsar import Pulsar

from enterprise.signals import parameter, gp_signals, white_signals, signal_base, selections
from enterprise.signals import gp_priors, utils
from enterprise_extensions import models, hypermodel, blocks, model_utils
import argparse
import pickle
from natsort import natsorted
import la_forge.core as co
from ceffyl import densities
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--N_freqs', type=int, default=10,
                    help='number of frequencies to use; DEFAULT=10')
parser.add_argument('--N_samples', type=int, default=int(1e7),
                    help='number of samples to analyse; DEFAULT=1e7')
parser.add_argument('--outdir', type=str,
                    help='path to directory to save chains')
parser.add_argument('--idx', type=int,
                    help='combined index of psr and sim realisation e.g. realisation 1, psr 4 => idx = 45*1+4 = 49')
parser.add_argument('--inc_red', action='store_true', default=False,
                    help='flag to also model intrinsic red noise; set to true for GFL Lite; DEFAULT=False')
parser.add_argument('--red_components', type=int, help='number of red frequencies if --inc_red==True; DEFAULT=10',
                    default=10)
args = parser.parse_args()

# load pulsars from single realisation
datadir = f'/home/lambwg/GFL/middleton21/simulations/realisations/realisation_{int(args.idx/45)}/'
timfiles = natsorted(glob.glob(datadir+'*.tim'))
parfiles = natsorted(glob.glob(datadir+'*.par'))

psrs = []
for par, tim in zip(parfiles, timfiles):
    psrs.append(Pulsar(par, tim))

Tspan = model_utils.get_tspan(psrs)  # total observational timespan

# timing model
tm = gp_signals.MarginalizingTimingModel(use_svd=True)

# white noise
efac = parameter.Constant(1.0)
ef = white_signals.MeasurementNoise(efac=efac)

# gwb (no spatial correlations)
log10_rho_gw = parameter.Uniform(-15.2, -1,
                                 size=args.N_freqs)('log10_rho')
cpl = gp_priors.free_spectrum(log10_rho=log10_rho_gw)
crn = gp_signals.FourierBasisGP(cpl, components=args.N_freqs,
                                Tspan=Tspan, name='common')

# full model is sum of components
model = ef + tm + crn

if args.inc_red:  # if GFL Lite, add intrinsic red noise
    # intrinsic red noise
    log10_A = parameter.Uniform(-20, -11)
    gamma = parameter.Uniform(0, 7)

    pl = utils.powerlaw(log10_A=log10_A, gamma=gamma)
    irn = gp_signals.FourierBasisGP(spectrum=pl, components=args.red_components,
                                    Tspan=Tspan, name='red_noise')
    model += irn

# intialize PTA
pta = {0: signal_base.PTA(model(psrs[args.idx%45]))}

# instanciate a collection of models for optimised sampling
super_model = hypermodel.HyperModel(pta)

# Set up sampler
outdir = args.outdir+f'realisation_{int(args.idx/45)}/psr_{args.idx%45}/'
sampler = super_model.setup_sampler(resume=True, outdir=outdir,
                                    empirical_distr=None, human='wglamb17',
                                    sample_nmodel=False)

# sample
N = int(args.N_samples)  # one mega-sample!
x0 = super_model.initial_sample()

sampler.sample(x0, N, neff=int(1e7))  # SAMPLE!

# save chain as la_forge core
chain = co.Core(args.outdir+f'realisation_{int(args.idx/45)}/psr_{args.idx%45}/')
chain.save(args.outdir+f'realisation_{int(args.idx/45)}/psr_{args.idx%45}/chain.core')

# calculate bandwidths for this posterior ready to fit
bw = densities.DE_factory(args.outdir+f'realisation_{int(args.idx/45)}/psr_{args.idx%45}/',
                          recursive=False, pulsar_names=['temporary name'])
bws = np.array([bw.bandwidth(chain(rho)) for rho in chain.params 
                if 'rho' in rho])
np.save(args.outdir+f'realisation_{int(args.idx/45)}/psr_{args.idx%45}/bandwidths',
        bws)