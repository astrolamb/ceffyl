# Script to run an uncorrelated full-likelihood analysis
# William G. Lamb 2022

import pickle
import glob
import enterprise.signals.parameter as parameter
from enterprise.signals import utils, signal_base, selections, white_signals
from enterprise.signals import gp_signals
from enterprise.pulsar import Pulsar
from enterprise_extensions import model_utils, hypermodel, model_orfs
import la_forge.core as co
import argparse

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
# ensure no backend separation
selection = selections.Selection(selections.no_selection)

efac = parameter.Constant(1.0)  # white noise parameters
ef = white_signals.MeasurementNoise(efac=efac, selection=selection)

# timing model
tm = gp_signals.MarginalizingTimingModel(use_svd=True)

# intrinsic red noise
log10_A = parameter.Uniform(-20, -11)
gamma = parameter.Uniform(0, 7)

pl = utils.powerlaw(log10_A=log10_A, gamma=gamma)
irn = gp_signals.FourierBasisGP(spectrum=pl, components=args.red_components,
                                Tspan=Tspan)

# GW parameters (initialize with names here to use parameters in common across pulsars)
log10_A_gw = parameter.Uniform(-18, -12)('log10_A_gw')
gamma_gw = parameter.Uniform(0, 7)('gamma_gw')

cpl = utils.powerlaw(log10_A=log10_A_gw, gamma=gamma_gw)
gw = gp_signals.FourierBasisGP(spectrum=cpl, Tspan=Tspan,
                               name='gw', components=args.gw_components)

# full model is sum of components
model = ef + tm + gw + irn

# intialize PTA
pta = {0: signal_base.PTA([model(psr) for psr in psrs])}

# instanciate hypermodel for optimised mcmc sampling
super_model = hypermodel.HyperModel(pta)

# Set up sampler
outdir = args.outdir+f'realisation_{args.idx}'
sampler = super_model.setup_sampler(resume=True, outdir=outdir,
                                    empirical_distr=None, sample_nmodel=False,
                                    human='wglamb17')

# sample
N = int(1e7)  # one mega-sample!
x0 = super_model.initial_sample()

sampler.sample(x0, N)  # SAMPLE!

# save chain as binary file using la_forge
chain = co.Core(args.outdir+f'realisation_{args.idx}')
chain.save(args.outdir+f'realisation_{args.idx}/chain.core')
