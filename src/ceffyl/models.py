"""
A module containing various useful spectral models for use.

The spectral models are used to generate the PSD of a stochastic process.
The models are used in the `enterprise` package to generate the PSD of
a stochastic process.

Modified for use in ceffyl from github.com/nanograv/enterprise

Example:
    To generate a power-law PSD model, use the `powerlaw` function:
    >>> import numpy as np
    >>> import ceffyl.models as models
    >>> f = 10**np.logspace(-9, -7, 100)[:, np.newaxis]
    >>> Tspan = 10 * 365.25 * 86400  # 10 years in seconds
    >>> log10_A = -15
    >>> gamma = 5
    >>> psd = models.powerlaw(f, df=1/Tspan, log10_A=log10_A, gamma=gamma)

    The `psd` variable will contain the power-law PSD model evaluated at
    the frequencies in `f`.
"""
# imports
from types import MethodType
import numpy as np
from numpy.typing import NDArray
import enterprise.constants as const


def powerlaw(f: float|np.ndarray,
             df: float,
             log10_A: float|np.ndarray = -16,
             gamma: float|np.ndarray = 13/3) -> float|np.ndarray:
    """
    Power-law model. PSD amplitude at each frequency is a power-law.

    Parameters
    ----------
    f : NDArray
        array of frequencies
    df : float
        frequency resolution
    log10_A : float or NDArray
        log10 amplitude at reference frequency of f=1/yr
    gamma : float or NDArray
        spectral index of power-law
        
    Returns
    -------
    rho2 : NDArray
        computed spectrum, units = [s^2] (i.e. PSD * df)
    """

    return ((10**log10_A)**2/12.0/np.pi**2 * const.fyr**(gamma-3)
            * f**(-gamma) * df)


def turnover(f: NDArray,
             df: float,
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
    df : float
        frequency resolution
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
    return hcf**2/12/np.pi**2/f**3 * df


def broken_powerlaw(f: NDArray,
                    df: float,
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
    df : float
        frequency resolution
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
        computed spectrum, units = [s^2] (i.e. PSD * df)
    """
    hcf = (10 ** log10_A
           * (f / const.fyr) ** ((3 - gamma) / 2)
           * (1+(f/10**log10_fb)**(1/kappa)) ** (kappa * (gamma - delta) / 2))
    return hcf ** 2 / (12*np.pi**2 * f**3) * df


def broken_turnover(f: NDArray,
                    df: float,
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
    df : float
        frequency resolution
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
        computed spectrum, units = [s^2] (i.e. PSD * df)
    """
    
    hcf = (10 ** log10_A
           * (f / const.fyr) ** ((3 - gamma) / 2)
           * (1+(f/10**log10_fb)**(1/smooth)) ** (smooth * (gamma - delta) / 2)
           / (1 + (10 ** lf0 / f) ** kappa) ** beta)
    return hcf**2/12/np.pi**2/f**3 * df


def t_process(f: NDArray,
              df: float,
              alphas: NDArray,
              log10_A: float = -15.,
              gamma: float = 13/3) -> NDArray:
    """
    t-process model - PSD amplitude at each frequency is a fuzzy power-law.

    Parameters
    ----------
    f : NDArray
        array of frequencies
    df : float
        frequency resolution
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
        computed spectrum, units = [s^2] (i.e. PSD * df)
    """
    alphas = alphas[:, np.newaxis]
    return powerlaw(f, df, log10_A=log10_A, gamma=gamma) * alphas


def psd_t_process(f: NDArray,
                  df: float,
                  alphas: NDArray,
                  psd: MethodType = powerlaw,
                  log10_A: float = -15.,
                  gamma: float = 13/3,
                  psd_kwargs: dict = None) -> NDArray:
    """
    t-process model - PSD amplitude at each frequency is a fuzzy PSD.

    Parameters
    ----------
    f : NDArray
        array of frequencies
    df : float
        frequency resolution
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
        computed spectrum, units = [s^2] (i.e. PSD * df)
    """
    alphas = alphas[:, None]
    return psd(f, df, log10_A=log10_A, gamma=gamma, **psd_kwargs) * alphas


def t_process_adapt(f: NDArray,
                    df: float,
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
    df : float
        frequency resolution
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
        computed spectrum, units = [s^2] (i.e. PSD * df)
    """

    alpha_model = np.ones_like(f)
    alpha_model[int(np.rint(nfreq))] = alphas_adapt

    return powerlaw(f, df=df, log10_A=log10_A, gamma=gamma) * alpha_model


def turnover_knee(f: NDArray,
                  df: float,
                  log10_A: float = -15,
                  gamma: float = 13/3,
                  lfb: float = None,
                  lfk: float = None,
                  kappa: float = 10/3,
                  delta: float = 5):
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
    return hcf ** 2 / 12 / np.pi ** 2 / f ** 3 * df


def free_spectrum(f, df, log10_rho):
    """
    Free spectral model. PSD  amplitude at each frequency is a free parameter.
    Model is parameterized by S(f_i) = \rho_i^2 * T, where \rho_i is the free
    parameter and T is the observation length

    Parameters
    ----------
    f : NDArray
        array of frequencies
    df : float
        frequency resolution
    log10rho : NDArray
        PSD amplitude at each frequency

    Returns
    -------
    rho2 : NDArray
        computed spectrum, units = [s^2] (i.e. PSD * df)
    """
    return 10 ** (2*log10_rho)


def powerlaw_genmodes(f: NDArray,
                      df: float,
                      log10_A: float = -16,
                      gamma: float = 5,
                      wgts: NDArray = None):
    """
    Power-law model with generalization to allow for different weights on each
    frequency bin.

    Parameters
    ----------
    f : NDArray
        array of frequencies
    df : float
        frequency resolution
    log10_A : float
        log10 amplitude at reference frequency of f=1/yr
    gamma : float
        spectral index of power-law
    wgts : NDArray, optional
        array of weights on the power-law at each frequency.
        If None, assumes all weights are 1.

    Returns
    -------
    rho2 : NDArray
        computed spectrum, units = [s^2] (i.e. PSD * df)
    """

    if wgts is not None:
        df = wgts ** 2
    return ((10 ** log10_A)**2 / 12.0 / np.pi ** 2
            * const.fyr ** (gamma - 3) * f ** (-gamma) * df)
