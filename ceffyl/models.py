"""
models.py
A module containing various useful PSD models for use in other modules.

Modified for use in ceffyl from github.com/nanograv/enterprise
"""
# imports
import numpy as np
from numpy.typing import NDArray
import enterprise.constants as const
from types import MethodType


def powerlaw(f: NDArray,
             Tspan: float,
             log10_A: float = -15.,
             gamma: float = 13/3) -> NDArray:
    """
    Basic powerlaw

    Parameters
    ----------
    f : NDArray
        array of frequencies
    Tspan : float
        timespan of dataset
    log10_A : float
        log10 amplitude at reference frequency of f=1/yr
    gamma : float
        spectral index

    Returns
    -------
    rho2 : NDArray
        computed spectrum, units = [s^2] (i.e. PSD/Tspan)
    """

    return ((10**log10_A)**2/12.0/np.pi**2 * const.fyr**(gamma-3)
            * f**(-gamma) / Tspan)


def turnover(f: NDArray,
             Tspan: float,
             log10_A: float = -15.,
             gamma: float = 13/3,
             lf0: float = -8.5,
             kappa: float = 10/3,
             beta: float = 0.5) -> NDArray:
    """
    Turnover model - effectively two powerlaws that smoothly transition

    Parameters
    ----------
    f : NDArray
        array of frequencies
    Tspan : float
        timespan of dataset
    log10_A : float
        log10 amplitude at reference frequency of f=1/yr
    gamma : float
        spectral index of higher-frequency powerlaw
    lf0 : float
        log10 of bend frequency
    kappa : float
        spectral index of low-frequency powerlaw
    beta : float
        parameter to control smoothing of powerlaw connection

    Returns
    -------
    rho2 : NDArray
        computed spectrum, units = [s^2]
    """

    hcf = (10 ** log10_A
           * (f / const.fyr) ** ((3 - gamma) / 2)
           / (1 + (10 ** lf0 / f) ** kappa) ** beta)
    return hcf**2/12/np.pi**2/f**3 / Tspan


def broken_powerlaw(f: NDArray,
                    Tspan: float,
                    log10_A: float = -15.,
                    gamma: float = 13/3,
                    delta: float = 0.1,
                    log10_fb: float = -9.,
                    kappa: float = 0.) -> NDArray:
    """
    Generic broken powerlaw spectrum

    Parameters
    ----------
    f : NDArray
        array of frequencies
    Tspan : float
        timespan of dataset
    log10_A : float
        log10 amplitude at reference frequency of f=1/yr
    gamma : float
        spectral index of low-frequency powerlaw
    log10_fb : float
        log10 of bend frequency between low-freq and high-freq powerlaws
    delta : float
        spectral index of high-frequency powerlaw
    kappa : float
        parameter to control smoothing between low-freq and high-freq powerlaws

    Returns
    -------
    rho2 : NDArray
        computed spectrum, units = [s^2] (i.e. PSD/Tspan)
    """
    hcf = (10 ** log10_A
           * (f / const.fyr) ** ((3 - gamma) / 2)
           * (1+(f/10**log10_fb)**(1/kappa)) ** (kappa * (gamma - delta) / 2))
    return hcf ** 2 / (12*np.pi**2 * f**3) / Tspan


def broken_turnover(f: NDArray,
                    Tspan: float,
                    log10_A: float = -15.,
                    gamma: float = 13/3,
                    lf0: float = -8.5,
                    kappa: float = 10/3,
                    beta: float = 0.5,
                    log10_fb: float = -9.,
                    delta: float = 0.,
                    smooth: float = 0.1) -> NDArray:
    """
    'Broken turnover' - three powerlaws!!

    Parameters
    ----------
    f : NDArray
        array of frequencies
    Tspan : float
        timespan of dataset
    log10_A : float
        log10 amplitude at reference frequency of f=1/yr
    gamma : float
        spectral index of mid-frequency powerlaw
    lf0 : float
        log10 of bend frequency between low-freq and mid-freq powerlaws
    kappa : float
        spectral index of low-frequency powerlaw
    beta : float
        parameter to control smoothing between low-freq and mid-freq powerlaws
    log10_fb : float
        log10 of bend frequency between mid-freq and high-freq powerlaws
    delta : float
        spectral index of high-frequency powerlaw
    smooth : float
        parameter to control smoothing between mid-freq and high-freq powerlaws

    Returns
    -------
    rho2 : NDArray
        computed spectrum, units = [s^2] (i.e. PSD/Tspan)
    """
    
    hcf = (10 ** log10_A
           * (f / const.fyr) ** ((3 - gamma) / 2)
           * (1+(f/10**log10_fb)**(1/smooth)) ** (smooth * (gamma - delta) / 2)
           / (1 + (10 ** lf0 / f) ** kappa) ** beta)
    return hcf**2/12/np.pi**2/f**3 / Tspan


def t_process(f: NDArray,
              Tspan: float,
              alphas: NDArray,
              log10_A: float = -15.,
              gamma: float = 13/3) -> NDArray:
    """
    t-process model - PSD amplitude at each frequency is a fuzzy power-law.

    Parameters
    ----------
    f : NDArray
        array of frequencies
    Tspan : float
        timespan of dataset
    log10_A : float
        log10 amplitude at reference frequency of f=1/yr
    gamma : float
        spectral index of mid-frequency powerlaw
    alphas : NDArray
        array of weights on the power-law at each frequency
        NOTE: assume alphas is an array with the same size as f

    Returns
    -------
    rho2 : NDArray
        computed spectrum, units = [s^2] (i.e. PSD/Tspan)
    """
    alphas = alphas[:, None]
    return powerlaw(f, Tspan, log10_A=log10_A, gamma=gamma) * alphas


def psd_t_process(f: NDArray,
                  Tspan: float,
                  alphas: NDArray,
                  psd: MethodType = powerlaw,
                  log10_A: float = -15.,
                  gamma: float = 13/3,
                  psd_kwargs: dict = {}) -> NDArray:
    """
    t-process model - PSD amplitude at each frequency is a fuzzy PSD.

    Parameters
    ----------
    f : NDArray
        array of frequencies
    Tspan : float
        timespan of dataset
    log10_A : float
        log10 amplitude at reference frequency of f=1/yr
    gamma : float
        spectral index of mid-frequency powerlaw
    alphas : NDArray
        array of weights on the power-law at each frequency
        NOTE: assume alphas is an array with the same size as f
    psd : MethodType
        psd function to modify
    psd_kwargs : dict, optional
        some keyword arguments for given psd

    Returns
    -------
    rho2 : NDArray
        computed spectrum, units = [s^2] (i.e. PSD/Tspan)
    """
    alphas = alphas[:, None]
    return psd(f, Tspan, log10_A=log10_A, gamma=gamma, **psd_kwargs) * alphas


def t_process_adapt(f: NDArray,
                    Tspan: float,
                    nfreq: float,
                    alphas_adapt: float,
                    log10_A: float = -15.,
                    gamma: float = 13.3) -> NDArray:
    """
    t-process model. PSD amplitude at *one* frequency is a fuzzy power-law

    Parameters
    ----------
    f : NDArray
        array of frequencies
    Tspan : float
        timespan of dataset
    nfreq : float
        index of parameter to apply weighting
    alphas_adapt : float
        magnitude of the weighting
    log10_A : float
        log10 amplitude at reference frequency of f=1/yr
    gamma : float
        spectral index of mid-frequency powerlaw

    Returns
    -------
    rho2 : NDArray
        computed spectrum, units = [s^2] (i.e. PSD/Tspan)
    """

    alpha_model = np.ones_like(f)
    alpha_model[int(np.rint(nfreq))] = alphas_adapt

    return powerlaw(f, Tspan, log10_A=log10_A, gamma=gamma) * alpha_model


def free_spectrum(f: NDArray,
                  Tspan: float,
                  log10_rho: NDArray) -> NDArray:
    """
    Free spectral model. PSD  amplitude at each frequency is a free parameter.
    Model is parameterized by S(f_i) = \rho_i^2 * T, where \rho_i is the free
    parameter and T is the observation length

    Parameters
    ----------
    f : NDArray
        array of frequencies
    Tspan : float
        timespan of dataset
    log10rho : NDArray
        PSD amplitude at each frequency

    Returns
    -------
    rho2 : NDArray
        computed spectrum, units = [s^2] (i.e. PSD/Tspan)
    """
    return 10 ** (2*log10_rho)
