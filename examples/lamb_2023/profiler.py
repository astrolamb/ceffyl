# Script to profile enterprise/ceffyl code
# William G. Lamb 2022

import numpy as np
import pickle
import glob
import json
import time
import sys
from enterprise.pulsar import Pulsar
from enterprise_extensions.models import model_2a, model_3a
import time
from enterprise.pulsar import Pulsar
from enterprise_extensions.models import model_2a, model_3a
from enterprise_extensions import model_utils, model_orfs
import enterprise.signals.parameter as parameter
from enterprise.signals import utils, signal_base, selections, white_signals
from enterprise.signals import gp_signals
from ceffyl import Ceffyl, models
import cpuinfo
print(cpuinfo.get_cpu_info()['brand_raw'])

# ----------------------------------------
# ENTERPRISE PTA OBJECT CREATION
# load psr timing/parameter files
datadir = f'~/ceffyl/data/sim51/'
timfiles = sorted(glob.glob(datadir+'timfiles/*.tim'))
parfiles = sorted(glob.glob(datadir+'parfiles/*.par'))

psrs = []  # create psr objects
for par, tim in zip(parfiles, timfiles):
    psrs.append(Pulsar(par, tim))

Tspan = model_utils.get_tspan(psrs)  # calculate dataset timespan

# list of pulsar names 
pulsar_names = np.loadtxt(datadir+'pulsar_list.txt',
                          dtype=np.unicode_)

# create PTA model
tm = gp_signals.MarginalizingTimingModel(use_svd=False)  # timing model

# white noise signals
efac = parameter.Constant(1.0)
ef = white_signals.MeasurementNoise(efac=efac,
                                    selection=selections.Selection(selections.no_selection))

# intrinsic red noise
log10_A = parameter.Uniform(-20, -11)
gamma = parameter.Uniform(0, 7)
pl = utils.powerlaw(log10_A=log10_A, gamma=gamma)
irn = gp_signals.FourierBasisGP(spectrum=pl, components=10,
                                Tspan=Tspan, name='red_noise')

# GW powerlaw parameters
log10_A_gw = parameter.Uniform(-18, -12)('log10_A_gw')
gamma_gw = parameter.Uniform(0, 7)('gamma_gw')
cpl = utils.powerlaw(log10_A=log10_A_gw, gamma=gamma_gw)
gw = gp_signals.FourierBasisGP(spectrum=cpl, Tspan=Tspan,
                               name='gw', components=10)  # common process
gw_hd = gp_signals.FourierBasisCommonGP(spectrum=cpl, Tspan=Tspan,
                                        name='gw', components=10,
                                        orf=model_orfs.hd_orf())  # HD SGWB

# creating PTA models
m2a = ef + tm + gw + irn  # uncorrelated powerlaw common process
m3a = ef + tm + gw_hd + irn  # HD-correlated SGWB powerlaw

# initialise empty lists
output = './profiler/'
m2a_avg, m3a_avg, gl_avg, gfl_avg, fs_avg = [], [], [], [], []
m2a_std, m3a_std, gl_std, gfl_std, fs_std = [], [], [], [], []

for ii in range(1, len(pulsar_names)+1):  # loop through pulsars!
    print(f'Loop {ii}\n')
    # initialize PTA
    pta_2a = signal_base.PTA([m2a(psr) for psr in psrs[:ii]])
    pta_3a = signal_base.PTA([m3a(psr) for psr in psrs[:ii]])
    
    # profiling uncorrelated PTA
    x0 = np.hstack(p.sample() for p in pta_2a.params)
    t0 = time.perf_counter()  # start timer
    pta_2a.get_lnlikelihood(x0)  # initial run to set up matrices
    [pta_2a.get_lnlikelihood(x0) for ii in range(50)]  # REPEAT!
    t1 = (time.perf_counter() - t0)/50  # mean time
    m2a_avg.append(t1)

    # profiling correlated PTA
    x0 = np.hstack(p.sample() for p in pta_3a.params)
    t0 = time.perf_counter()  # start timer
    pta_3a.get_lnlikelihood(x0)  # initial run to set up matrices
    [pta_3a.get_lnlikelihood(x0) for ii in range(50)]  # REPEAT!
    t1 = (time.perf_counter() - t0)/50  # mean time
    m3a_avg.append(t1)
    #--------------------------
    
    # profiling GFL Lite
    # load data
    gfllite = Ceffyl.ceffyl(datadir + 'GFL_lite_sim51/')
    gw = Ceffyl.signal(N_freqs=10, name='gw', selected_psrs=pulsar_names[:ii])  # create CP signal
    gfllite.add_signals([gw])  # add signal to model

    x0 = gfllite.initial_samples()  # initial sample
    t0 = time.perf_counter()  # start timer
    [gfllite.ln_likelihood(x0) for ii in range(10000)]  # REPEAT!
    t1 = (time.perf_counter() - t0)/10000  # mean time
    gl_avg.append(t1)
    #--------------------------
    
    # profiling GFL
    # load data
    gfl = Ceffyl.ceffyl(datadir + 'GFL_sim51/')
    irn = Ceffyl.signal(N_freqs=10, common_process=False, name='red_noise',
                     params=[parameter.Uniform(-20, -11)('log10_A'),
                             parameter.Uniform(0, 7)('gamma')],
                     selected_psrs=pulsar_names[:ii])  # intrinsic red noise signal
    gfl.add_signals([irn, gw])  # add irn + cp signals
    
    x0 = gfl.initial_samples()  # initial sample
    t0 = time.perf_counter()  # start timer
    [gfl.ln_likelihood(x0) for ii in range(10000)]  # REPEAT!
    t1 = (time.perf_counter() - t0)/10000  # mean time
    gfl_avg.append(t1)
    #--------------------------

# profile free spectrum refit
fs = Ceffyl.ceffyl(datadir + 'freespec_sim51/')
gw = Ceffyl.signal(N_freqs=10, name='gw')  # add signal
fs.add_signals([gw])

x0 = fs.initial_samples()  # initial sample
t0 = time.perf_counter()  # start timer
[fs.ln_likelihood(x0) for ii in range(10000)]  # REPEAT!
t1 = (time.perf_counter() - t0)/10000  # mean time
fs_avg.append(t1)

# save the data
np.save('./profiles/m2a', m2a_avg)
np.save('./profiles/m3a', m3a_avg)
np.save('./profiles/gl', gl_avg)
np.save('./profiles/gfl', gfl_avg)
np.save('./profiles/fs', fs_avg)