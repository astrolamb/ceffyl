"""
A module for defining priors for the Ceffyl model.

This module contains classes and functions to create and manage priors
for the parameters of the Ceffyl model. It is designed to be used with the
scipy.stats module, or numpyro's distributions module for probabilistic
programming.


"""

# imports
import numpy as np
from scipy.stats import uniform, norm
from jax.scipy.stats import uniform as jax_uniform, norm as jax_norm
