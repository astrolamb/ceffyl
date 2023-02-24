"""
A utilities module containing various useful
functions for use in other modules.

Modified for use in ceffyl from github.com/nanograv/enterprise
"""

import numpy as np
import enterprise.constants as const


def powerlaw(f, Tspan, log10_A=-16, gamma=5):

    return ((10**log10_A)**2/12.0/np.pi**2 * const.fyr**(gamma-3)
            * f**(-gamma) / Tspan)


def turnover(f, Tspan, log10_A=-15, gamma=13/3,
             lf0=-8.5, kappa=10/3, beta=0.5):

    hcf = (10 ** log10_A
           * (f / const.fyr) ** ((3 - gamma) / 2)
           / (1 + (10 ** lf0 / f) ** kappa) ** beta)
    return hcf**2/12/np.pi**2/f**3 / Tspan


def broken_turnover(f, Tspan, log10_A=-15, gamma=13/3, lf0=-8.5, kappa=10/3,
                    beta=0.5, log10_fb=-9, delta=0.1, smooth=0.1):
    hcf = (10 ** log10_A
           * (f / const.fyr) ** ((3 - gamma) / 2)
           * (1+(f/10**log10_fb)**(1/smooth)) ** (smooth * (gamma - delta) / 2)
           / (1 + (10 ** lf0 / f) ** kappa) ** beta)
    return hcf**2/12/np.pi**2/f**3 / Tspan


def t_process(f, Tspan, log10_A=-15, gamma=4.33, alphas=None):
    """
    t-process model. PSD  amplitude at each frequency
    is a fuzzy power-law.

    NOTE: assume alphas is an array with the same size as f
    """
    alphas = alphas[:, None]
    return powerlaw(f, Tspan, log10_A=log10_A, gamma=gamma) * alphas


def psd_t_process(f, Tspan, psd=powerlaw, log10_A=-15, gamma=5,
                  alphas=None, **psd_kwargs):
    """
    t-process model. PSD  amplitude at each frequency
    is a fuzzy psd.

    NOTE: assume alphas is an array with the same size as f
    """
    alphas = alphas[:, None]
    return psd(f, Tspan, log10_A=log10_A, gamma=gamma, **psd_kwargs) * alphas


def t_process_adapt(f, Tspan, log10_A=-15, gamma=4.33, alphas_adapt=None,
                    nfreq=None):
    """
    t-process model. PSD  amplitude at each frequency
    is a fuzzy power-law.
    """
    if alphas_adapt is None:
        alpha_model = np.ones_like(f)
    else:
        if nfreq is None:
            alpha_model = alphas_adapt
        else:
            alpha_model = np.ones_like(f)
            alpha_model[int(np.rint(nfreq))] = alphas_adapt

    return powerlaw(f, Tspan, log10_A=log10_A, gamma=gamma) * alpha_model


def turnover_knee(f, Tspan, log10_A=-15, gamma=5, lfb=None,
                  lfk=None, kappa=10/3, delta=5):
    """
    Generic turnover spectrum with a high-frequency knee.
    :param f: sampling frequencies of GWB
    :param A: characteristic strain amplitude at f=1/yr
    :param gamma: negative slope of PSD around f=1/yr (usually 13/3)
    :param lfb: log10 transition frequency at which environment dominates GWs
    :param lfk: log10 knee frequency due to population finiteness
    :param kappa: smoothness of turnover (10/3 for 3-body stellar scattering)
    :param delta: slope at higher frequencies
    """
    hcf = (
        10 ** log10_A
        * (f / const.fyr) ** ((3 - gamma) / 2)
        * (1.0 + (f / 10 ** lfk)) ** delta
        / np.sqrt(1 + (10 ** lfb / f) ** kappa)
    )
    return hcf ** 2 / 12 / np.pi ** 2 / f ** 3 / Tspan


def broken_powerlaw(f, Tspan, log10_A=-15, gamma=5, delta=0.1, log10_fb=-9,
                    kappa=0.1):
    """
    Generic broken powerlaw spectrum.
    :param f: sampling frequencies
    :param A: characteristic strain amplitude [set for gamma at f=1/yr]
    :param gamma: negative slope of PSD for f > f_break [set for comparison
        at f=1/yr (default 13/3)]
    :param delta: slope for frequencies < f_break
    :param log10_fb: log10 transition frequency at which slope switches from
        gamma to delta
    :param kappa: smoothness of transition (Default = 0.1)
    """
    hcf = (10 ** log10_A
           * (f / const.fyr) ** ((3 - gamma) / 2)
           * (1+(f/10**log10_fb)**(1/kappa)) ** (kappa * (gamma - delta) / 2))
    return hcf ** 2 / (12*np.pi**2 * f**3) / Tspan

def free_spectrum(f, Tspan, log10_rho):
    """
    Free spectral model. PSD  amplitude at each frequency is a free parameter.
    Model is parameterized by S(f_i) = \rho_i^2 * T, where \rho_i is the free
    parameter and T is the observation length
    """
    return 10 ** (2*log10_rho)

def powerlaw_genmodes(f, Tspan, log10_A=-16, gamma=5, wgts=None):
    if wgts is not None:
        df = wgts ** 2
    else:
        df = 1/Tspan
    return ((10 ** log10_A)**2 / 12.0 / np.pi ** 2
            * const.fyr ** (gamma - 3) * f ** (-gamma) * df)


def infinitepower(f, Tspan):
    return np.full_like(f, 1e40, dtype="d")
