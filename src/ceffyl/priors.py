"""
A module for defining priors for the Ceffyl model.

This module contains classes and functions to create and manage priors
for the parameters of the Ceffyl model. It is designed to be used with the
scipy.stats module, or numpyro's distributions module for probabilistic
programming.
"""

# imports
from scipy.stats import uniform, norm

try:
    from numpyro.distributions import (Uniform as NumpyroUniform,
                                       Normal as NumpyroNormal)
    NumpyroUniform.logpdf = NumpyroUniform.log_prob
    NumpyroUniform.rvs = NumpyroUniform.sample
    NumpyroUniform.ppf = NumpyroUniform.icdf
    NumpyroNormal.logpdf = NumpyroNormal.log_prob
    NumpyroNormal.rvs = NumpyroNormal.sample
    NumpyroNormal.ppf = NumpyroNormal.icdf
except ImportError:
    pass

class Uniform:
    """
    A class to define a uniform distribution for use as a prior in the Ceffyl
    model.

    Args:
        low: The lower bound of the uniform distribution.
        high: The upper bound of the uniform distribution.
        name: The name of the distribution.

    Methods:
        logpdf: A method to calculate the log probability density function of
                the uniform distribution at a given value.
        sample: A method to sample from the uniform distribution.

    """
    def __init__(self, low: float, high: float, name: str = None,
                 size: int = 1, numpyro: bool = False):
        self.low = low
        self.high = high
        self.name = name
        self.numpyro = numpyro
        self.size = size
        
        if numpyro:
            self.dist = NumpyroUniform(low=low, high=high, name=name)
        else:
            self.dist = uniform(loc=low, scale=high-low)

    def logpdf(self, x):
        return self.dist.logpdf(x)

    def sample(self):
        return self.dist.rvs(size=self.size)

class Normal:
    """
    A class to define a normal distribution for use as a prior in the Ceffyl
    model.

    Args:
        loc: The mean of the normal distribution.
        scale: The standard deviation of the normal distribution.
        name: The name of the distribution.

    Methods:
        logpdf: A method to calculate the log probability density function of
                the normal distribution at a given value.
        sample: A method to sample from the normal distribution.

    """
    def __init__(self, loc: float, scale: float, name: str = None,
                 size: int = 1, numpyro: bool = False):
        self.loc = loc
        self.scale = scale
        self.name = name
        self.numpyro = numpyro
        self.size = size
        
        if numpyro:
            self.dist = NumpyroNormal(loc=loc, scale=scale, name=name)
        else:
            self.dist = norm(loc=loc, scale=scale)

    def logpdf(self, x):
        return self.dist.logpdf(x)

    def sample(self):
        return self.dist.rvs(size=self.size)
