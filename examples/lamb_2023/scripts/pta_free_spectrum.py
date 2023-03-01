# Script to create a PTA free spectrum model
# William G. Lamb 2022

import glob
from enterprise.pulsar import Pulsar
from enterprise.signals import parameter, gp_signals, white_signals, signal_base, selections, utils
from enterprise.signals import gp_priors as gpp
from enterprise_extensions import models, hypermodel, blocks, model_utils
import argparse
import pickle
import la_forge.core as co

parser = argparse.ArgumentParser()
parser.add_argument('--outdir', type=str,
                    help='specify directory to save chains')
parser.add_argument('--idx', type=int,
                    help='index of the simulation realisation')
parser.add_argument('--red_components', type=int,
                    help='number of intrinsic red noise frequencies; DEFAULT=10',
                    default=10)
parser.add_argument('--gw_components', type=int,
                    help='number of GW frequencies; DEFAULT=10',
                    default=10)
args = parser.parse_args()

# path to the data set you want to analyse
datadir = f'~/ceffyl/data/sim_{args.idx}/'
timfiles = sorted(glob.glob(datadir+'*.tim'))
parfiles = sorted(glob.glob(datadir+'*.par'))

psrs = []  # create enterprise pulsar objects
for par, tim in zip(parfiles, timfiles):
    psrs.append(Pulsar(par, tim))

Tspan = model_utils.get_tspan(psrs)  # calculate observation timespan

##### parameters and priors #####
# timing model
tm = gp_signals.MarginalizingTimingModel(use_svd=True)

# white noise signals
efac = parameter.Constant(1.0)
ef = white_signals.MeasurementNoise(efac=efac,
                                    selection=selections.Selection(selections.no_selection))

# intrinsic red noise
log10_A = parameter.Uniform(-20, -11)
gamma = parameter.Uniform(0, 7)

pl = utils.powerlaw(log10_A=log10_A, gamma=gamma)
irn = gp_signals.FourierBasisGP(spectrum=pl, components=args.red_components,
                                Tspan=Tspan, name='red_noise')

# GW parameters (initialize with names here to use parameters in common across pulsars)
log10_rho = parameter.Uniform(-12.2, -2, size=args.gw_components)('log10_rho')

cpl = gpp.free_spectrum(log10_rho=log10_rho)
gw = gp_signals.FourierBasisGP(spectrum=cpl, Tspan=Tspan,
                               name='gw', components=args.gw_components)

model = tm + ef + irn + gw

pta = signal_base.PTA([model(p) for p in psrs])

# intialize PTA
pta = {0: pta}

# instanciate a collection of models
super_model = hypermodel.HyperModel(pta)

# Set up sampler
outdir = args.outdir+f'realisation_{args.idx}'
sampler = super_model.setup_sampler(resume=False, outdir=outdir,
                                    empirical_distr=None, sample_nmodel=False,
                                    human='wglamb17')

# sample
N = int(1e7)  # one mega-sample!
x0 = super_model.initial_sample()

sampler.sample(x0, N)  # SAMPLE!

# save chain as a binary la_forge core
chain = co.Core(args.outdir+f'realisation_{args.idx}')
chain.save(args.outdir+f'realisation_{args.idx}/chain.core')